import sys
import string

input_path = sys.argv[1]
train_data_path = input_path + "/train.txt"
dev_data_path = input_path + "/valid.txt"
test_data_path = input_path + "/test.txt"
data_paths = {"train": train_data_path, "valid": dev_data_path, "test": test_data_path}
data_path = sys.argv[2]

def convert(filepath):
    with open(filepath, encoding="utf-8") as f:
        guid = 0
        tokens = []
        pos_tags = []
        chunk_tags = []
        ner_tags = []
        types = []
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if tokens:
                    yield guid, {
                        "id": str(guid),
                        "tokens": tokens,
                        "pos_tags": pos_tags,
                        "chunk_tags": chunk_tags,
                        "ner_tags": ner_tags,
                        "tokens_type": types,
                    }
                    guid += 1
                    tokens = []
                    pos_tags = []
                    chunk_tags = []
                    ner_tags = []
                    types = []
            else:
                # conll2003 tokens are space separated
                splits = line.split(" ")
                tokens.append(splits[0])
                pos_tags.append(splits[1])
                chunk_tags.append(splits[2])
                ner_tags.append(splits[3].rstrip())
                topen_type = splits[3].rstrip()
                if topen_type != "O":
                    types.append(topen_type)
                else:
                    types.append(splits[0])
        # last example
        yield guid, {
            "id": str(guid),
            "tokens": tokens,
            "pos_tags": pos_tags,
            "chunk_tags": chunk_tags,
            "ner_tags": ner_tags,
            "tokens_type": types,
        }

for name, filepath in data_paths.items():
    data = convert(filepath)
    sens = []
    for exp in data:
        #print(exp)
        #raise Exception("Print an example.")
        sen = " ".join(exp[1]["tokens"])
        sens.append(sen)
    with open(data_path+"/"+name+".txt","w") as wf:
        wf.write(" ".join(sens))

for name, filepath in data_paths.items():
    data = convert(filepath)
    token_with_types = []
    for exp in data:
        sen = " ".join(exp[1]["tokens_type"])
        token_with_types.append(sen)
    with open(data_path+"/"+name+"_type.txt","w") as wf:
        wf.write(" ".join(token_with_types))