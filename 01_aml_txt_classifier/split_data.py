import os, random, csv, sys
import numpy as np

def message_generator(acl_dir):
    # download from here: http://ai.stanford.edu/~amaas/data/sentiment/
    # cite: http://ai.stanford.edu/~amaas/papers/wvSent_acl2011.bib
    train_dir = acl_dir + '/train'
    test_dir = acl_dir + '/test'
    
    for set_dir in [train_dir, test_dir]:
        pos_dir = set_dir + '/pos'
        neg_dir = set_dir + '/neg'
        
        for annotation, review_dir in enumerate([neg_dir, pos_dir]):
            print('\t ' + review_dir)
            for fname in os.listdir(review_dir):
                message = open(review_dir + '/' + fname, 'r', encoding='utf-8').read().strip().replace('<br />', '\n')
                msg_id = fname[:fname.index('.txt')]
                yield (msg_id, message, annotation)

                
def get_split(data_dir, train_perc):
    '''
    Returns the ACL IMDB data, split into a train and a test set.
    
    train_perc: specifies the percentage data to use for training. 
                Must be a float greater than 0 and less than 1.
                The rest of the data is used for testing.
    data_dir:   the directory for the extracted ACL IMDB data
    '''

    assert train_perc > 0 and train_perc < 1
    
    train = []
    test = []
    
    review_idx = 1 # create an ID number for each review
    
    print("Getting train and testing split...")
    for review in message_generator(data_dir):
        choice = np.random.choice([True, False], 1, p=[train_perc, 1 - train_perc])[0]
        if choice:
            train.append(review)
        else:
            test.append(review)
    
    print("Found", len(train), "train datapoints and", len(test), "test datapoints.")
    random.shuffle(train)
    random.shuffle(test)
    
    return train, test
    
    
def save_data(data, fname):
    '''
    Saves an input list of samples into a CSV file.
    
    data:  the list of samples, each in triplets of: (review_id, message, annotation)
    fname: the CSV file name to save the data to.
    '''
    writer = csv.writer(open(fname, 'w', encoding='utf-8'))
    writer.writerow(['id', 'message', 'annotation'])
    for item in data:
        writer.writerow([item[0], item[1].replace('\n', ' '), item[2]])
        
    print("Saved", len(data), "samples to file:", fname)


def main():
    # data was split into 50% train, 50% test, but we want more training data, 
    # so we will go instead for a 90/10 split. 
    train, test = get_split(sys.argv[1], float(sys.argv[2]))
    save_data(train, sys.argv[3])
    save_data(test, sys.argv[4])


if __name__ == '__main__':
    main()