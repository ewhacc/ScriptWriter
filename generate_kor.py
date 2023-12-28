#!/usr/bin/env python
# coding: utf-8

# # Install

# In[1]:


# cuda 11.1
#!pip install -r requirements-torch-cu111.txt --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111


# In[2]:


#!pip list | grep -E "gdown"


# In[3]:


#import gdown


# In[4]:


prefix = 'final'
# AI hub 업로드 용 변경
#url = 'https://drive.google.com/uc?id=1qQQbjFd0c7unV-S18t-7x3KUw5OFVl3P'


# In[5]:


#scripts_file = f'data/json_split.zip'
#gdown.download(url, scripts_file, quiet=False)


# In[6]:


#!zipu --extract --encoding cp949 'data/json_split.zip' 'data/'


# In[7]:


get_ipython().system('du -sh data/train')
get_ipython().system('du -sh data/validation')
#!du -sh data/test


# In[8]:


from glob import glob
import json


# In[9]:


json_files = sorted(glob('data/train/*/*.json'))


# In[10]:


len(json_files)


# In[11]:


data_dict = []
for json_file in json_files:
    with open(json_file) as f:
        story_dict = json.load(f)
    units = story_dict['units']
    for unit in units:
        unit_dict = {}
        unit_dict['uid'] = unit['id']
        unit_dict['storyline'] = unit['storyline']
        unit_dict['script'] = []
        for story_script in unit['story_scripts']:
            unit_dict['script'].append(story_script['content'])
        data_dict.append(unit_dict)


# In[12]:


train_num = len(data_dict)
print('train num =', train_num)


# In[13]:


json_files = sorted(glob('data/validation/*/*.json'))
print(len(json_files))


# In[14]:


val_num = 0
for json_file in json_files:
    with open(json_file) as f:
        story_dict = json.load(f)
    units = story_dict['units']
    for unit in units:
        unit_dict = {}
        unit_dict['uid'] = unit['id']
        unit_dict['storyline'] = unit['storyline']
        unit_dict['script'] = []
        for story_script in unit['story_scripts']:
            unit_dict['script'].append(story_script['content'])
        data_dict.append(unit_dict)
        val_num += 1
print('val_num =', val_num)


# In[15]:


json_files = sorted(glob('data/test/*/*.json'))
print(len(json_files))


# In[16]:


test_num = 0
for json_file in json_files:
    with open(json_file) as f:
        story_dict = json.load(f)
    units = story_dict['units']
    for unit in units:
        unit_dict = {}
        unit_dict['uid'] = unit['id']
        unit_dict['storyline'] = unit['storyline']
        unit_dict['script'] = []
        for story_script in unit['story_scripts']:
            unit_dict['script'].append(story_script['content'])
        data_dict.append(unit_dict)
        test_num += 1
print('test num =', test_num)


# In[17]:


len(data_dict)


# # Data preperation

# In[18]:


#prefix = 'final'
#url = 'https://drive.google.com/uc?id=1Bts2h-QPQ5-m7sDIXgVRfumjl-8XHOst'
#url = 'https://drive.google.com/uc?id=1x6HuyJTQcNydJ9P-fJl2LtxnnAu9Vp8N'
#prefix = '1cycle'
#url = 'https://drive.google.com/uc?id=1j46elyFZtkmnmCehlntMi0eX0Tp5nnav'
#prefix = 'helper'
#url = 'https://drive.google.com/uc?id=1iSP_YKFs56d5cRRTEMzfedwRxrx-nXWO'


# ## 스토리헬퍼 샘플 데이터 다운로드

# In[19]:


#scripts_file = f'data/scripts_{prefix}.json'
#zip_file = f'data/scripts_{prefix}.zip'
#gdown.download(url, zip_file, quiet=False)


# In[20]:


#!unzip $zip_file 
#!mv -f 'final.json' $scripts_file


# In[21]:


#import json

#with open(scripts_file) as f:
#    data_dict = json.load(f)


# In[22]:


# 샘플 데이터 출력
data_dict[0]


# **후처리**
# 1. `\n`을 제거한다. "부엌에서 일하게 된 마리오\n인부들 사이에서 인기만점인 베아트리체"  
#    ==> 필요없는 것 같음.

# In[23]:


# 비정상적 white character가 없는지 확인
for idx, data in enumerate(data_dict):
    #data['storyline'] = data['storyline'].replace('\n', ' ')
    for i, context in enumerate(data['script']):
        #if '\n' in context:
        if '부엌에서 일하게' in context:
            print(idx, i, context)
            print('"%s%s"'%(context[9],context[10]))
            print(context[10] == ' ')
        #data['script'][i] = context.replace('\n', ' ')


# ## Tokenizer

# ### kobigbird pretrained model을 이용한 tokenize

# In[24]:


from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('monologg/kobigbird-bert-base')


# # Word2Vec

# In[25]:


import numpy as np

EOS_ID = tokenizer.sep_token_id

positive_sessions = []
positive_str = []
positive_ids = []
for i, unit_data in enumerate(data_dict):
    unit_contexts = [tokenizer.tokenize(text) for text in unit_data['script'] + ['[SEP]'] ]
    while [] in unit_contexts:
        print('empty string in the script. removing..., id=', i)
        index = unit_contexts.index([])
        #print("'{%s}'"%unit_data['script'][index])
        del unit_contexts[index]
        #unit_contexts.remove([])
        del unit_data['script'][index]
    if len(unit_contexts) <= 1:
        print('empty scripts. skipping..., id=', i)
        continue
    unit_narrative = tokenizer.tokenize(unit_data['storyline'])
    if len(unit_narrative) == 0:
        print('empty narrative. skipping, id=', i)
        continue
    positive_sessions.append([unit_contexts, unit_narrative, 1])
    positive_str.append(unit_data)
    positive_ids.append(unit_data['uid'])
print("all suitable sessions: ", len(positive_sessions))

# reproducibility를 위한 random seed 설정
np.random.seed(42)

# split policy 변경으로 shuffle 안함
# random shuffle data
#np.random.shuffle(positive_sessions)
#np.random.seed(42)
#np.random.shuffle(positive_str)
#np.random.seed(42)
#np.random.shuffle(positive_ids)


# In[26]:


positive_ids[0:10]


# In[27]:


#train_num = int(len(positive_sessions) * 0.9)
#dev_test_num = int(len(positive_sessions) * 0.05)
train_sessions, dev_sessions, test_sessions = positive_sessions[:train_num], positive_sessions[train_num: train_num + val_num], positive_sessions[train_num + val_num:]
print('number of train =', len(train_sessions), ', val =', len(dev_sessions), ', test =', len(test_sessions))


# In[28]:


train_texts = []
for train_session in train_sessions:
    train_texts += train_session[0]
    train_texts.append(train_session[1])
print('number of word2vec training sentences =', len(train_texts))


# In[29]:


from gensim.models import Word2Vec

# word2vec 학습
model = Word2Vec(sentences = train_texts, vector_size = 200, window = 7, min_count = 5, workers = 4)


# In[30]:


print('total num of words =', len(model.wv.key_to_index))
print('first word = "%s"'%model.wv.index_to_key[0])
print('last word = "%s"'%model.wv.index_to_key[-1])


# In[31]:


# word2vec이 잘 학습되었는지 여러가지 테스트를 수행하자.
print(model.wv.most_similar("가족"))


# ## 데이터 저장
# 
# `embeddings.pkl`과 `vocab.txt`를 생성한다.

# In[32]:


with open(f"data/vocab_{prefix}.txt", "w", encoding="utf-8") as file:
    for i, key in enumerate(model.wv.index_to_key):
        file.write('%s\t%i\n'%(key, i))


# In[33]:


import pickle

new_embeddings = np.array([[0.]*200],dtype='float32') 
for i in range(len(model.wv.index_to_key)):
    new_embeddings = np.append(new_embeddings, [model.wv.get_vector(i)], axis=0)

with open(f'data/embeddings_{prefix}.pkl', 'wb') as f:
    pickle.dump(new_embeddings, f)


# In[34]:


model.save(f"data/word2vec_{prefix}.model")


# # 학습 데이터셋 준비

# In[35]:


EOS_ID = model.wv.key_to_index['[SEP]']+1
UNK_ID = model.wv.key_to_index['[UNK]']+1


# In[36]:


import pickle

vocab = {}

with open(f"data/vocab_{prefix}.txt", "r", encoding="utf-8") as fr:
    for idx, line in enumerate(fr):
        line = line.strip().split("\t")
        vocab[line[0]] = idx + 1


# In[37]:


# sample id 출력 확인
vocab['가족']


# **positive data 준비**

# In[38]:


positive_data = []
positive_str2 = []

for unit, unit_str in zip(positive_sessions, positive_str):
    narrative = unit[1]
    #print(narrative)
    context = unit[0]
    narrative_id = [vocab.get(word, UNK_ID) for word in narrative]
    context_id = [[vocab.get(word, UNK_ID) for word in sent] for sent in context]
    if len(narrative_id) == 0 or len(context_id) == 0:
        print('empty narrative found. skipping...')
        #print(unit[0])
        #print(unit[1])
        print(unit_str)
        continue
    data = [context_id, narrative_id, 1]
    positive_data.append(data)
    positive_str2.append(unit_str)


# In[39]:


len(positive_str), len(positive_str2), len(positive_ids)


# In[40]:


train, dev, test = positive_data[:train_num], positive_data[train_num: train_num + val_num], positive_data[train_num + val_num:]
train_ids, dev_ids, test_ids = positive_ids[:train_num], positive_ids[train_num: train_num + val_num], positive_ids[train_num + val_num:]


# In[41]:


import random
train_all, dev_all, test_all = [], [], []
for context_id, narrative_id, _ in train:
    num_context = len(context_id)
    for i in range(1, num_context):
        context = context_id[:i]
        response = context_id[i]
        train_all.append([context, response, narrative_id, response, 1])
        flag = True
        while flag:
            random_idx = random.randint(0, len(positive_data) - 1)
            random_context = positive_data[random_idx][0]
            random_idx_2 = random.randint(0, len(random_context) - 1)
            random_response = random_context[random_idx_2]
            if len(response) != len(random_response):
                flag = False
                train_all.append([context, random_response, narrative_id, response, 0])
            else:
                for idx, wid in enumerate(response):
                    if wid != random_response[idx]:
                        flag = False
                        train_all.append([context, random_response, narrative_id, response, 0])
                        break
print(train_all[0]) 
print(train_all[1])


# In[42]:


dev_all_ids = []
for i_dev, (context_id, narrative_id, _) in enumerate(dev):
    num_context = len(context_id)
    for i in range(1, num_context):
        context = context_id[:i]
        response = context_id[i]
        dev_all.append([context, response, narrative_id, response, 1])
        dev_all_ids.append(dev_ids[i_dev])
        count = 0
        negative_samples = []
        # fix count 버그
        while count < 8:
            random_idx = random.randint(0, len(positive_data) - 1)
            random_context = positive_data[random_idx][0]
            random_idx_2 = random.randint(0, len(random_context) - 1)
            random_response = random_context[random_idx_2]
            if random_response not in negative_samples and random_response != [EOS_ID]:
                if len(response) != len(random_response):
                    dev_all.append([context, random_response, narrative_id, response, 0])
                    negative_samples.append(random_response)
                    dev_all_ids.append(dev_ids[i_dev])
                    count += 1
                else:
                    for idx, wid in enumerate(response):
                        if wid != random_response[idx]:
                            dev_all.append([context, random_response, narrative_id, response, 0])
                            negative_samples.append(random_response)
                            dev_all_ids.append(dev_ids[i_dev])
                            count += 1
                            break
        if response == [EOS_ID]:
            dev_all.append([context, [EOS_ID], narrative_id, response, 1])
        else:
            dev_all.append([context, [EOS_ID], narrative_id, response, 0])
        dev_all_ids.append(dev_ids[i_dev])
print(dev_all[0], dev_all[1], dev_all[2])


# In[43]:


test_all = []
test_all_ids = []
test_num_context = []
for i_test, (context_id, narrative_id, _) in enumerate(test):
    num_context = len(context_id)
    test_num_context.append(num_context-1)
    for i in range(1, num_context):
        context = context_id[:i]
        response = context_id[i]
        test_all.append([context, response, narrative_id, response, 1])
        test_all_ids.append(test_ids[i_test])
        count = 0
        negative_samples = []
        # fix count 버그
        while count < 8:
            random_idx = random.randint(0, len(positive_data) - 1)
            random_context = positive_data[random_idx][0]
            random_idx_2 = random.randint(0, len(random_context) - 1)
            random_response = random_context[random_idx_2]
            if random_response not in negative_samples and random_response != [EOS_ID]:
                if len(response) != len(random_response):
                    test_all.append([context, random_response, narrative_id, response, 0])
                    negative_samples.append(random_response)
                    test_all_ids.append(test_ids[i_test])
                    count += 1
                else:
                    for idx, id in enumerate(response):
                        if id != random_response[idx]:
                            test_all.append([context, random_response, narrative_id, response, 0])
                            negative_samples.append(random_response)
                            test_all_ids.append(test_ids[i_test])
                            count += 1
                            break
        if response == [EOS_ID]:
            test_all.append([context, [EOS_ID], narrative_id, response, 1])
        else:
            test_all.append([context, [EOS_ID], narrative_id, response, 0])
        test_all_ids.append(test_ids[i_test])
if test_num > 0:
    print(test_all[0], test_all[1], test_all[2])


# In[44]:


len(test_all_ids), len(test_all)


# In[45]:


print('total train count =', len(train_all))
print('total val count =', len(dev_all))
print('total test count =', len(test_all))


# In[46]:


np.sum(np.array(test_num_context))


# In[47]:


def get_numpy_from_nonfixed_2d_array(aa, max_sentence_len=50, max_num_utterance=10, padding_value=0):
    PAD_SEQUENCE = np.array([0] * max_sentence_len)
    rows = np.empty([0, max_sentence_len], dtype='int')
    aa = aa[-max_num_utterance:]
    for a in aa:
        sentence_len = len(a)
        if sentence_len < max_sentence_len:
            rows  = np.append(rows, [np.pad(a, (0, max_sentence_len-sentence_len), 'constant', constant_values=padding_value)[:max_sentence_len]], axis=0)
        else:
            rows = np.append(rows, [a[:max_sentence_len]], axis=0)
    num_utterance = len(aa)
    if num_utterance < max_num_utterance:
        rows = np.append(rows, [PAD_SEQUENCE]*(max_num_utterance-num_utterance), axis=0)
    # add empty +1 sentence
    rows = np.append(rows, [PAD_SEQUENCE], axis=0)
    #return np.concatenate(rows, axis=0).reshape(-1, max_sentence_len)
    return rows

def get_numpy_from_nonfixed_1d_array(a, max_sentence_len=50, padding_value=0):
    sentence_len = len(a)
    if sentence_len < max_sentence_len:
        return np.pad(a, (0, max_sentence_len-sentence_len), 'constant', constant_values=padding_value)
    else:
        return np.array(a[:max_sentence_len])

cc_test_data = [
        [1,2],
        [4,5,6],
        [7]
     ]
#get_numpy_from_nonfixed_2d_array(cc_test_data, max_sentence_len=5, max_num_utterance=4)


# In[ ]:


#try:
#    __IPYTHON__
#    from tqdm.notebook import tqdm
#except NameError:
#    from tqdm import tqdm
try:
    __IPYTHON__
    import sys
    if 'ipykernel' in sys.modules:
        pass
    elif 'IPython' in sys.modules:
        raise
    else:
        raise
    from tqdm.notebook import tqdm
except:
    from tqdm import tqdm
    
def pad_process(data, max_sentence_len=50, max_num_utterance=10):
    utterance = []
    response = []
    narrative = []
    gt_response = []
    y_true = []
    for unit in tqdm(data):
        utterance.append(get_numpy_from_nonfixed_2d_array(unit[0]))
        response.append(get_numpy_from_nonfixed_1d_array(unit[1]))
        narrative.append(get_numpy_from_nonfixed_1d_array(unit[2]))
        gt_response.append(get_numpy_from_nonfixed_1d_array(unit[3]))
        y_true.append(unit[4])
    utterance = np.stack(utterance)
    response = np.stack(response)
    narrative = np.stack(narrative)
    gt_response = np.stack(gt_response)
    y_true = np.stack(y_true)
    return (utterance, response, narrative, gt_response, y_true)

train_pad = pad_process(train_all)
dev_pad = pad_process(dev_all)
if test_num > 0:    
    test_pad = pad_process(test_all)
else:
    test_pad = ([], [], [], [], [])


# **학습데이터셋 저장**

# In[ ]:


with open(f'data/train_{prefix}.pkl', 'wb') as f:
    pickle.dump(train_pad, f)
with open(f'data/dev_{prefix}.pkl', 'wb') as f:
    pickle.dump(dev_pad, f)
with open(f'data/test_{prefix}.pkl', 'wb') as f:
    pickle.dump(test_pad, f)


# In[ ]:


with open(f'data/positive_{prefix}.pkl', "wb") as f:
    pickle.dump(positive_data, f)
with open(f'data/positive_str_{prefix}.pkl', "wb") as f:
    pickle.dump(positive_str2, f)


# In[ ]:


with open(f'data/test_all_ids_{prefix}.pkl', "wb") as f:
    pickle.dump(test_all_ids, f)


# In[ ]:


get_ipython().system('rm -rf ~/.cache/huggingface/datasets/story_data')


# In[ ]:


for unit, unit_str in zip (positive_data, positive_str2):
    len_unit = len(unit[0])
    len_unit_str = len(unit_str['script'])
    if len_unit != len_unit_str+1:
        print(len_unit, len_unit_str)
    #print(unit[0])
    #print(unit_str['script'])
    #break


# In[ ]:


def get_dat(index, data_pad, ids = None):
    utterances = data_pad[0][index]
    response = data_pad[1][index]
    narrative = data_pad[2][index]
    gt_response = data_pad[3][index]
    y_true = data_pad[4][index]
    narrative = narrative[narrative!=0]
    response = response[response!=0]
    gt_response = gt_response[gt_response!=0]
    #print([model.wv.index_to_key[k-1] for k in narrative])
    narrative_str = tokenizer.convert_tokens_to_string([model.wv.index_to_key[k-1] for k in narrative])
    response_str = tokenizer.convert_tokens_to_string([model.wv.index_to_key[k-1] for k in response])
    gt_response_str = tokenizer.convert_tokens_to_string([model.wv.index_to_key[k-1] for k in gt_response])
    #print(y_true)
    utterance_str = ['']*10
    for i in range(10):
        utterance = utterances[i]
        utterance = utterance[utterance!=0]
        if len(utterance) == 0:
            break
        utterance_str[i] =  tokenizer.convert_tokens_to_string([model.wv.index_to_key[k-1] for k in utterance])
    #print()
    if ids == None:
        id_ = None
    else:
        id_ = ids[index]
    return id_, narrative_str, response_str, gt_response_str, y_true, utterance_str
    
def browse_dat(index, data_pad):
    utterances = data_pad[0][index]
    response = data_pad[1][index]
    narrative = data_pad[2][index]
    gt_response = data_pad[3][index]
    y_true = data_pad[4][index]
    narrative = narrative[narrative!=0]
    response = response[response!=0]
    gt_response = gt_response[gt_response!=0]
    #print([model.wv.index_to_key[k-1] for k in narrative])
    print('N:', tokenizer.convert_tokens_to_string([model.wv.index_to_key[k-1] for k in narrative]))
    print('R:', tokenizer.convert_tokens_to_string([model.wv.index_to_key[k-1] for k in response]))
    print('T:', tokenizer.convert_tokens_to_string([model.wv.index_to_key[k-1] for k in gt_response]))
    print(y_true)
    for i in range(10):
        utterance = utterances[i]
        utterance = utterance[utterance!=0]
        if len(utterance) == 0:
            break
        print('U:', tokenizer.convert_tokens_to_string([model.wv.index_to_key[k-1] for k in utterance]))
    print()
    

#browse_dat(0, train_pad)


# In[ ]:


if test_num > 0:
    for i in range(110,120): 
        browse_dat(i, test_pad)


# In[ ]:


import pandas as pd


# In[ ]:


column_names = ['id', 'Narrative', 'Response', 'GT_Response', 'y_true', 'score', 'R2@1', 'R10@1', 'R10@2', 'R10@5', 'MRR', 'AVG']
for i in range(10):
    column_names.append('U%02d'%(i+1))
print(column_names)
df = pd.DataFrame(columns=column_names)


# In[ ]:


n = len(test_all_ids)
data_dict_all = []
for i in tqdm(range(n)):
    id_, narrative_str, response_str, gt_response_str, y_true, utterance_str = get_dat(i, test_pad, test_all_ids)
    data_dict = { }
    data_dict['id'] = id_
    data_dict['Narrative'] = narrative_str
    data_dict['Response'] = response_str
    data_dict['GT_Response'] = gt_response_str
    data_dict['y_true'] = y_true
    for i in range(10):
        data_dict[f'U%02d'%(i+1)] = utterance_str[i]
    #new_row = pd.Series(data_dict)
    #df = pd.concat([df, new_row.to_frame().T], ignore_index=True)
    data_dict_all.append(data_dict)


# In[ ]:


df = pd.DataFrame.from_dict(data_dict_all)


# In[ ]:


df


# In[ ]:


# output rows are too large for excel
# save to csv
df.to_csv(f'test_output_{prefix}.csv', index=False)


# In[ ]:


column_names = ['id', 'Narrative', 'Response', 'GT_Response', 'y_true', 'score', 'R2@1', 'R10@1', 'R10@2', 'R10@5', 'MRR', 'AVG']
for i in range(10):
    column_names.append('U%02d'%(i+1))
print(column_names)
df = pd.DataFrame(columns=column_names)


# In[ ]:


n = len(dev_all_ids)
data_dict_all = []
for i in tqdm(range(n)):
    id_, narrative_str, response_str, gt_response_str, y_true, utterance_str = get_dat(i, dev_pad, dev_all_ids)
    data_dict = { }
    data_dict['id'] = id_
    data_dict['Narrative'] = narrative_str
    data_dict['Response'] = response_str
    data_dict['GT_Response'] = gt_response_str
    data_dict['y_true'] = y_true
    for i in range(10):
        data_dict[f'U%02d'%(i+1)] = utterance_str[i]
    #new_row = pd.Series(data_dict)
    #df = pd.concat([df, new_row.to_frame().T], ignore_index=True)
    data_dict_all.append(data_dict)


# In[ ]:


df = pd.DataFrame.from_dict(data_dict_all)
df.to_csv(f'dev_output_{prefix}.csv', index=False)


# In[ ]:




