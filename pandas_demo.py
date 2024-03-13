import pandas as pd

data = pd.DataFrame({'Yes':[50,21], 'No':[131, 23]})
data = pd.DataFrame({'Bob': ['I liked it.', 'It was awful.'],
             'Sue': ['Pretty good.', 'Bland.']})

data = pd.DataFrame({'Bob': ['I liked it.', 'It was awful.'],
                     'Sue': ['Pretty good.', 'Bland.']},
                    index=['Product A', 'Product B'])
print(data)

series = pd.Series([1, 2, 3, 4, 5])
series = pd.Series([30, 35, 40], index=['2015 Sales',
                   '2016 Sales', '2017 Sales'], name='Product A')
print(series)

print(data.dtypes)
print(data.Bob.astype('string'))
print(data.index.dtype)

data.rename(columns={'points': 'score'})
data.rename(index={0: 'firstEntry', 1: 'secondEntry'})
data.rename_axis("wines", axis='rows').rename_axis("fields", axis='columns')

canadian_youtube = pd.read_csv("../input/youtube-new/CAvideos.csv")
british_youtube = pd.read_csv("../input/youtube-new/GBvideos.csv")
pd.concat([canadian_youtube, british_youtube])


left = canadian_youtube.set_index(['title', 'trending_date'])
right = british_youtube.set_index(['title', 'trending_date'])

left.join(right, lsuffix='_CAN', rsuffix='_UK')
