from pandas import read_excel
from pathlib import Path

def read_csv():
    my_sheet = 'BatchImport' 
    file_name = '01_Oakley_batch_import.xlsx'
    df = read_excel(Path('data',file_name), sheet_name = my_sheet,keep_default_na=False)
    cleaned = clean(df)
    data= cleaned.to_dict('index') #each value in data is one instant of a product. (tags and description)
    return data

def clean(df):
    to_keep=["brand","name","madein","category","subcategory","season",
            "color","bicolors","gender"
            #,"description-it","description-en",
            #"description-fr","description-de","description-es","description-ro"
            #,"description-nl","description-pl","description-pt","description-cs",
            #"description-sk","description-sv","description-hu",	"description-et"
            #,"description-ru","description-bg","description-da","description-fi",
            #"description-lt","description-el"
            ]
    to_drop=[]
    for col in df.columns:
        if col not in to_keep:
            to_drop.append(col)
    df.drop(to_drop, inplace=True, axis=1)
    print(len(df.index))
    return df

def write_as_txt(data):
    c=0
    for k,v in data.items():
        path = "/Users/niyoush/test_set/"
        with open(path+"product"+str(c)+".txt", 'w') as f:
            print(v, file=f)
        print(c)
        c+=1
    

write_as_txt(read_csv())
