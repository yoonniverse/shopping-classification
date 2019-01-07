import pickle
import numpy as np
import re

counts_list = []
price_list = []

for i in range(9):
    with open('info/info-{}.pkl'.format(i+1), 'rb') as f:
        temp = pickle.load(f)
        counts_list.append(temp[0])
        price_list.append(temp[1])

# Aggergate information

counts = counts_list[0]
for i in range(1, 9):
    counts = counts.add(counts_list[i], fill_value=0)

price = np.mean(price_list)

# Disregard useless words and special characters from `counts`

useless_words = ['', '[불명]']
regex = '[상품|상세|설명|기타|없음|참조]|[^가-힣a-zA-Z]'

usable_words = set(counts.index) - set(useless_words)
usable_words_counts = counts.loc[usable_words]
usable_words_counts = usable_words_counts.loc[[idx for idx in usable_words_counts.index if not re.findall(regex, idx)]].sort_values(ascending=False)

# Select top frequency words with thresh

thresh = 10000
usable_words = list(usable_words_counts.index[:thresh])

print(len(usable_words))
print(usable_words[:100])
print(price)

with open('info.pkl', 'wb') as handle:
    pickle.dump((usable_words, price), handle)
