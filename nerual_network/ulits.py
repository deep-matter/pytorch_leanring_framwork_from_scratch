import os
import unicodedata
import torch
import glob
import random
import io
import string


all_letters = string.ascii_letters + ",.;'"
num_letters = len(all_letters)


# stringVal = u'áæãåāœčćęßßßわた'

# print(unicodedata.normalize('NFKD', stringVal).encode('ascii', 'ignore').decode())
# print(num_letters)



def unicode_ascii(list_string):
    corrected_string = ''
    for i in range(0,len(list_string)):
        applied_unicode = unicodedata.normalize(
            "NFKD", list_string[i]).encode('ascii', 'ignore').decode()
        # print(applied_unicode)
        corrected_string += applied_unicode
    return corrected_string


# correted_sample = unicode_ascii(all_letters)
# print(correted_sample)
#########################
#itertions through files and folders 

directory_files= 'dataset/data/names' # name

  

def find_path(path):
    for diro , subdiro , filename  in os.walk(path):
        full_paths = []
        for files in filename : 
            full_path =  diro + os.path.sep + files
            if full_path.endswith('.txt'):
                full_paths.append(full_path)
            else :
                print('done')  
    return full_paths

    #     print(os.path.join(directory_files,file)) 

    

def read_lines_of_file(filename):
    open_file = io.open(filename, encoding='utf-8').read().strip().split('\n')  
    return [unicode_ascii(line) for line in open_file ]

#########################

def loading_data():
    category_line = {}
    all_categories =[]

    for filename in find_path(directory_files):
        category_language = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category_language)

        read_lines = read_lines_of_file(filename)
        category_line[category_language] = read_lines
    return category_line , all_categories         

def letter_to_index(letter):
    return all_letters.find(letter)

def letter_to_tensor(letter):
        tensor = torch.zeros(1,num_letters)
        tensor[0][letter_to_index(letter)] = 1 
        return tensor 

def line_to_tensor(line):
    tensor_line = torch.zeros(len(line) , 1 , num_letters)
    for i , letter in enumerate(line):
        tensor_line[i][0][letter_to_index(letter)]=1
        return tensor_line     


def random_trainig_samples(category_line,all_categories ):
    
    def random_chose_train(sample_num):
        tensor_size_of_sample = torch.randint(0 ,len(sample_num), -1)
        return sample_num[tensor_size_of_sample]

    category = random_chose_train(all_categories)
    line = random_chose_train(category_line[category])    
    category_tensor = torch.tensor([all_categories.index(category)],dtype=torch.long)
    line_tensor =   line_to_tensor(line)

    return category , line , category_tensor. line_category

if __name__ == '__main__':
    pass

    ######### intialize the data training 
    