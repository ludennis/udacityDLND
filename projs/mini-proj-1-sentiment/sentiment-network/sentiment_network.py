import numpy as np
import time
import sys

from collections import Counter

class SentimentNetwork:
    '''
    Fields/attributes:
        - weights (random)
        - learning rate
        - num of input nodes
        - num of hidden nodes
        - num of output nodes
    Functions:
        - preprocess data
        - train
            - forward pass
            - backpropagation
        - test
        - run

    Neural Network
        - input: all the counts of words in a review
        - output: NEGATIVE or POSITIVE
    '''
        
    def init_network(self,num_input_nodes,num_hidden_nodes,num_output_nodes,learn_rate):
        self.num_input_nodes = num_input_nodes
        self.num_hidden_nodes = num_hidden_nodes
        self.num_output_nodes = num_output_nodes
        
        self.learn_rate = learn_rate

        self.weights_i_h = np.zeros((self.num_input_nodes,self.num_hidden_nodes))
        self.weights_h_o = np.random.normal(0.0, self.num_output_nodes**-0.5, (self.num_hidden_nodes,self.num_output_nodes))

    def set_learn_rate(self,learn_rate):
        self.learn_rate = learn_rate

    def preprocess_data(self,reviews,labels,min_count=1,polarity_cutoff=0):
        '''
        TODO: add two variables:
            1. min_count
            2. polarity_cutoff
        '''

        word_count = Counter()
        pos_word_count = Counter()
        neg_word_count = Counter()

        for review,label in zip(reviews,labels):
            for word in review.split(' '):
                word_count[word] += 1
                if label == 'POSITIVE':
                    pos_word_count[word] += 1
                elif label == 'NEGATIVE':
                    neg_word_count[word] += 1

        temp_set = set()
        for word, count in word_count.items():
            ratio = np.log(pos_word_count[word]/float(neg_word_count[word]+1))
            if ((count >= min_count) and abs(ratio) >= polarity_cutoff):
                temp_set.add(word)

        self.word_vocab = {}
        for i,word in enumerate(list(temp_set)):
            self.word_vocab[word] = i

        self.label_vocab = {}
        temp_set = set([label for label in labels])
        for i,label in enumerate(list(temp_set)):
            self.label_vocab[label] = i

    def sigmoid(self,x):
        return (1 / (1 + np.exp(-x)))

    def sigmoid_prime(self,x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def label_to_num(self,label):
        return 1 if label=='POSITIVE' else 0

    def train(self,reviews,labels):
        assert(len(reviews) == len(labels))

        start_time = time.time()
        correct_so_far = 0

        np.random.seed(1)

        # input_layer = np.zeros((1,self.num_input_nodes))
        self.weights_i_h = np.zeros((self.num_input_nodes,self.num_hidden_nodes))
        self.weights_h_o = np.random.normal(0.0, self.num_output_nodes**-0.5, (self.num_hidden_nodes,self.num_output_nodes))

        hidden_layer = np.zeros((1,self.num_hidden_nodes))

        # pre-process reviews so it's not (1,74074) 
        print ('Pre-processing reviews for training ...')
        processed_reviews = []
        for review in reviews:
            input_layer_indices = set()
            for word in review.split(' '):
                if (word in self.word_vocab.keys()):
                    input_layer_indices.add(self.word_vocab[word])
            processed_reviews.append(list(input_layer_indices))

        print ('Training with learn rate: {} ...'.format(self.learn_rate))

        for i,(review_indices,label) in enumerate(zip(processed_reviews,labels)):
            # review = [1, num_input_nodes]
            # label = [1,1]

            #update input
            
            # input_layer *= 0
            hidden_layer *= 0

            # forward pass
            # no actviation for hidden layer
            # sigmoid activation function for output layer
            # input nodes will have the count of word in each review
            # inputs = (1, 74064) weights = (74064,10) hidden = (1,10)
            # hidden_layer (1,10) / weights[index] (,10)
            
            for index in review_indices:
                hidden_layer[0] += self.weights_i_h[index]
            # hidden_layer = np.dot(input_layer,self.weights_i_h)
            hidden_layer_a = hidden_layer

            output_layer = np.dot(hidden_layer_a, self.weights_h_o)
            output_layer_a = self.sigmoid(output_layer)

            # back propagate            
            # error (1,1) dot output_layer (1,1) = (1,1)
            output_error = np.array(output_layer_a - self.label_to_num(label)).reshape(1,1)
            error_term_output = np.dot(output_error,self.sigmoid_prime(output_layer))

            # (1,10) = error_term_output (1,1) x weights_h_o.T (1,10) 
            error_term_hidden = np.dot(error_term_output,self.weights_h_o.T) * 1
            
            # weights_h_o (10,1)
            # error_term_output (1,1)
            # hidden_layer_a (1,10)
            self.weights_h_o -= np.dot(hidden_layer_a.T,error_term_output) * self.learn_rate

            # weights_i_h (74064,10)
            # input_layer (1,74064) / input_layer.T (74064,1)
            # error_term_hidden (1,10)
            # input_layer[0][index] = 1
            for index in review_indices:
                self.weights_i_h[index] -= 1 * error_term_hidden[0] * self.learn_rate
            # self.weights_i_h -= np.dot(input_layer.T,error_term_hidden) * self.learn_rate

            #correct prediction
            if (output_layer_a >= 0.5 and label == 'POSITIVE') or \
               (output_layer_a < 0.5 and label == 'NEGATIVE'):
                correct_so_far += 1

            elapsed_time = time.time() - start_time
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0

            sys.stdout.write('\rProgress: {:.1f} % Speed(reviews/sec): {:.5f} #Correct: {} #Trained: {} Training Accuracy: {:.4f}%' \
                              .format(100 * i/float(len(reviews)),
                                      reviews_per_second,
                                      correct_so_far,
                                      i+1,
                                      correct_so_far * 100 / float(i+1)))
            if(i % 2500 == 0): print ('')
        print('')


    def test(self,reviews,labels):
        ''' test the trained neural network against inputted reviews and labels
            with forward pass
        '''
        assert (len(reviews) == len(labels))

        start_time = time.time()
        correct_so_far = 0
        hidden_layer = np.zeros((1,self.num_hidden_nodes))
        output_layer = np.zeros((1,self.num_output_nodes))

        for i,(review,label) in enumerate(zip(reviews,labels)):
            
            #init network before forward pass
            input_layer_indices = []
            for word in review.split(' '):
                if (word in self.word_vocab.keys()):
                    input_layer_indices.append(self.word_vocab[word])
            input_layer_indices = list(set(input_layer_indices))

            hidden_layer *= 0
            output_layer *= 0

            #forward pass to find prediction
            for index in input_layer_indices:
                hidden_layer[0] += self.weights_i_h[index]

            hidden_layer_a = hidden_layer

            # hidden_layer_a (1,10) 
            # weights_h_o (10,1) 
            output_layer = np.dot(hidden_layer_a,self.weights_h_o)
            output_layer_a = self.sigmoid(output_layer)

            if (output_layer_a >= 0.5 and label == 'POSITIVE') or \
               (output_layer_a < 0.5 and label == 'NEGATIVE'):
               correct_so_far += 1

            sys.stdout.write('\rProgress: {}% Speed(test/sec): {} #Correct: {} Accuracy: {:2f}%'\
                            .format(100*float(i)/len(reviews),
                                    float(i)/(time.time()-start_time),
                                    correct_so_far,
                                    correct_so_far/float(i+1)))
        print('')    

            


            

