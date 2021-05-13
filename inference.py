import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

import data
import model as model_file
import model_ori_with_type 
import data2 as data_ori_type

from utils import batchify, get_batch, repackage_hidden

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--debug', type=int, default=-1,
                    help='location of the data corpus')
parser.add_argument('--data', type=str, default='data/recipe_ori',
                    help='location of the data corpus')
parser.add_argument('--data_type', type=str, default='data/recipe_type',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN, GRU)')
parser.add_argument('--emsize', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1150,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=30,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=8000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.3,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.65,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.5,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--tied', action='store_false',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str,  default='RCP_or_type.pt', #RCP_LSTM_ori_with_type
                    help='path to save the final model')
parser.add_argument('--save_type', type=str,  default='RCP_type_LSTM_one_vocab.pt',
                    help='path to save the final model')
parser.add_argument('--alpha', type=float, default=2,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
args = parser.parse_args()



mcq_wrd = ['chicken','bread', 'apple', 'milk', 'salt', 'tomato'] #ch=6134, bread=3553, apple = 16, milk=4359, salt=10576, tomato=3965
#mcq_ids = [192, 398, 1437, 41, 70, 740]

# record = {corpus.dictionary.word2idx['chicken'] : [], corpus.dictionary.word2idx['bread'] : [], corpus.dictionary.word2idx['apple'] : [], corpus.dictionary.word2idx['milk'] : [], corpus.dictionary.word2idx['salt'] : [], corpus.dictionary.word2idx['tomato'] : []}
record = {192:[], 398:[], 1437:[], 41:[], 70:[], 740:[] }
mcq_result = {192:[], 398:[], 1437:[], 41:[], 70:[], 740:[] }
# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.set_device(0)
        torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

corpus = data_ori_type.Corpus(args.data)

eval_batch_size = 10
test_batch_size = 1
train_data = batchify(corpus.train, args.batch_size, args)
val_data = batchify(corpus.valid, eval_batch_size, args)
test_data = batchify(corpus.test, test_batch_size, args)

train_data_type = batchify(corpus.train_type, args.batch_size, args)
val_data_type = batchify(corpus.valid_type, eval_batch_size, args)
test_data_type = batchify(corpus.test_type, test_batch_size, args)


corpus2 = data.Corpus(args.data_type)


train_data2 = batchify(corpus2.train, args.batch_size, args)
val_data2 = batchify(corpus2.valid, eval_batch_size, args)
test_data2 = batchify(corpus2.test, test_batch_size, args)


###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
model = model_ori_with_type.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.tied)
if args.cuda:
    model.cuda()
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in model.parameters())
print('Args:', args)
print('Model total parameters:', total_params)


ntokens2 = len(corpus2.dictionary)
model2 = model_file.RNNModel(args.model, ntokens2, args.emsize, args.nhid, args.nlayers, args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.tied)
if args.cuda:
    model2.cuda()
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in model2.parameters())
print('Model total parameters:', total_params)



criterion = nn.CrossEntropyLoss()

# print (ntokens, ntokens2)

###############################################################################
# Testing code
###############################################################################

def evaluate(data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        output, hidden = model(data, hidden)
        output_flat = output.view(-1, ntokens)
        total_loss += len(data) * criterion(output_flat, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss.item() / len(data_source)

def evaluate2(data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model2.eval()
    if args.model == 'QRNN': model2.reset()
    total_loss = 0
    ntokens2 = len(corpus2.dictionary)
    hidden = model2.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        output, hidden = model2(data, hidden)
        output_flat = output.view(-1, ntokens2)
        total_loss += len(data) * criterion(output_flat, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss.item() / len(data_source)


def get_symbol_table(data, types):
    id_map ={}
    i = 0
    for pos, tp in zip(data, types):
        id_map.update({pos.data[0]:tp.data[0]})
    return id_map


def evaluate_both(data_source, data_source_type, data_source2, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model2.eval()
    model.eval()
    if args.model == 'QRNN': 
        model2.reset()
        model.reset()
    total_loss = 0
    total_loss2 = 0
    total_loss_cb = 0

    ntokens2 = len(corpus2.dictionary)
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    hidden2 = model2.init_hidden(batch_size)
    m = nn.Softmax()
    mcq_ids = [corpus.dictionary.word2idx[w] for w in mcq_wrd]


    for batch,i in enumerate(range(0, data_source.size(0) - 1, args.bptt)):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        data2, targets2 = get_batch(data_source2, i, args, evaluation=True)

        data_type, targets_type = get_batch(data_source_type, i, args, evaluation=True)

        

        if(batch_size==1):
            hidden = model.init_hidden(batch_size)
            hidden2 = model2.init_hidden(batch_size)

        output2, hidden2 = model2(data2, hidden2)
        output, hidden = model(data, data_type, hidden)

        output_flat2 = output2.view(-1, ntokens2)
        output_flat = output.view(-1, ntokens)

        
        candidates = set([corpus.dictionary.idx2word[i.data[0]] for i in targets])
        candidates_ids = set([i.data[0] for i in targets])

        candidates_type = set([corpus2.dictionary.idx2word[i.data[0]] for i in targets2])
        candidates_ids_type = set([i.data[0] for i in targets2])
        numwords = output_flat.size()[0]
        symbol_table = get_symbol_table(targets, targets2)



        output_flat_cb= output_flat.clone()
        sums = []
        for idxx in range(numwords):
            for pos in candidates_ids: #for all candidates

                tp = symbol_table[pos]
                var_prob = output_flat_cb.data[idxx][pos]
                type_prob = output_flat2.data[idxx][tp]
                new_prob1 = 2*var_prob #just to scale values, emperical
                
                if corpus.dictionary.idx2word[pos]!=corpus2.dictionary.idx2word[tp]: new_prob1 = (var_prob + type_prob) #/ 2
                output_flat_cb.data[idxx][pos] = new_prob1
                
        total_loss += len(data) * criterion(output_flat, targets).data
        total_loss2 += len(data2) * criterion(output_flat2, targets2).data
        total_loss_cb += len(data) * criterion(output_flat_cb, targets).data
        
        
        #########
        temp_output = output_flat_cb.clone()
        # print("our model") 
        # or 
        temp_output_entity_composite = output_flat.clone()
        temp_output_type = output_flat2.clone()
        # print("awd-st baseline")
        #########


        val, keys_t = temp_output.data.max(1)

        val_entity_composite, keys_t_entity_composite = temp_output_entity_composite.data.max(1)
        val_type, keys_t_type = temp_output_type.data.max(1)

        prob_temp_output = m(temp_output)
        prob_temp_output_baseline = m(temp_output_entity_composite)
        prob_temp_output_type = m(temp_output_type)

        prb_val, prb_keys = prob_temp_output.data.max(1)
        prb_val_entity_composite, prb_keys_entity_composite = prob_temp_output_baseline.data.max(1)
        prb_val_type, prb_keys_type = prob_temp_output_type.data.max(1)

        for i in range(len(targets.data)): 
            w= targets.data[i]

            voilated = 0
            base = temp_output.data[i][w]
            if w in mcq_ids: 
                r = 0
                r2 = 0
                pred = keys_t[i]
                if pred==w: 
                    r=1
                for idd in mcq_ids:
                    if idd!=w:
                        if base<temp_output.data[i][idd]:
                            voilated=1
                            break
                     
                record[w].append(r)
                if voilated==0: r2 = 1
                mcq_result[w].append(r2)


            
            


        # print (' soccer: ', len(data) * criterion(output_flat, targets).data), ' my: ',  len(data) * criterion(output_flat_cb, targets).data
        if(batch%500==0): 
            # print(' only ingred not avg')
            # print ("done batch ", batch, ' of ', len(data_source)/ eval_batch_size)
            test_loss_cb = total_loss_cb[0] / len(data_source)
            test_loss = total_loss[0] / len(data_source)
            test_loss2 = total_loss2[0] / len(data_source)
            p = (100*batch)/(33000)
            print('=' * 160)
            print('| after: {:5.2f}% | test var loss {:5.2f} | test var ppl {:8.2f} | test type loss {:5.2f} | test type ppl {:8.2f} | test cb loss {:5.2f} | test cb ppl {:8.2f}'.format(
                p, test_loss, math.exp(test_loss), test_loss2, math.exp(test_loss2),  test_loss_cb, math.exp(test_loss_cb) ))
            print('=' * 160)

        hidden = repackage_hidden(hidden)
        hidden2 = repackage_hidden(hidden2) 


    for idd in record: 
        if len(record[idd]) >0:
            print (corpus.dictionary.idx2word[idd], ' acc: ', sum(record[idd]), ' out of ', len(record[idd]), sum(record[idd])*100.0/len(record[idd])  )
            print (corpus.dictionary.idx2word[idd], ' mcq acc: ', sum(mcq_result[idd]), ' out of ', len(mcq_result[idd]), sum(mcq_result[idd])*100.0/len(mcq_result[idd]))



    return total_loss[0] / len(data_source), total_loss2[0] / len(data_source2), total_loss_cb[0] / len(data_source)



# Load the best saved model.
with open(args.save, 'rb') as f:
    model.load_state_dict(torch.load(f))
with open(args.save_type, 'rb') as f:
    model2.load_state_dict(torch.load(f))

test_batch_size = 1

test_loss, test_loss2, test_loss_cb = evaluate_both(test_data, test_data_type, test_data2, test_batch_size)
print('=' * 165)
print('| End of testing | test var loss {:5.2f} | test var ppl {:8.2f} | test type loss {:5.2f} | test type ppl {:8.2f} | test cb loss {:5.2f} | test cb ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss), test_loss2, math.exp(test_loss2),  test_loss_cb, math.exp(test_loss_cb) ))
print('=' * 165)