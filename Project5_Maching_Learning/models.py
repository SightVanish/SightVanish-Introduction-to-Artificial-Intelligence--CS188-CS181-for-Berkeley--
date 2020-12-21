import nn
import time

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        # w is weights, initialized with 1
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return(nn.DotProduct(x, self.w))         

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        score = nn.as_scalar(nn.DotProduct(x, self.w))
        if score >= 0:
            return 1
        else:
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        converge = False
        while not converge:
            converge = True
            # go throught the dataset, x, y is Constant with shape 1x2 and 1x1
            for x, y in dataset.iterate_once(1):
                # check whether the predict is same with the label
                if self.get_prediction(x) == nn.as_scalar(y):
                    continue
                else:
                    # update weights
                    self.w.update(x, nn.as_scalar(y))
                    converge = False

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        # construct a 3-level net
        # output = relu(relu(x*w1+b1)*w2+b2)*w3+b3
        self.w1 = nn.Parameter(1, 100) # *w1--batch_size x 10
        self.b1 = nn.Parameter(1, 100)
        self.w2 = nn.Parameter(100, 100) # *w2--batch_size x 10
        self.b2 = nn.Parameter(1, 100)
        self.w3 = nn.Parameter(100, 1) # *w3--batch_size x 1
        self.b3 = nn.Parameter(1, 1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        x1 = nn.Linear(x, self.w1)
        x1_b1 = nn.AddBias(x1, self.b1)
        relu1 = nn.ReLU(x1_b1)
        x2 = nn.Linear(relu1, self.w2)
        x2_b2 = nn.AddBias(x2, self.b2)
        relu2 = nn.ReLU(x2_b2)
        x3 = nn.Linear(relu2, self.w3)
        x3_b3 = nn.AddBias(x3, self.b3)
        return x3_b3

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        # # set parameters
        multiplier = -0.01 # note: multiplier is different from learning rate, it must be negative
        batch_size = 2
        loss = 1
        # update
        while loss > 0.01: # note: if it is set to 0.02, the final loss may be larger
            for x, y in dataset.iterate_once(batch_size):
                loss = self.get_loss(x, y)
                grad_wrt_w1, grad_wrt_b1, grad_wrt_w2, grad_wrt_b2, grad_wrt_w3, grad_wrt_b3 = nn.gradients(loss, [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3])
                self.w1.update(grad_wrt_w1, multiplier)
                self.b1.update(grad_wrt_b1, multiplier)
                self.w2.update(grad_wrt_w2, multiplier)
                self.b2.update(grad_wrt_b2, multiplier)
                self.w3.update(grad_wrt_w3, multiplier)
                self.b3.update(grad_wrt_b3, multiplier)
                loss = nn.as_scalar(self.get_loss(x, y))

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        # construct a 3-level net
        # output = relu(relu(relu(x*w1+b1)*w2+b2)*w3+b3)*w4+b4
        self.w1 = nn.Parameter(784, 100) # *w1--batch_size x 100
        self.b1 = nn.Parameter(1, 100)
        self.w2 = nn.Parameter(100, 50) # *w2--batch_size x 50
        self.b2 = nn.Parameter(1, 50)
        self.w3 = nn.Parameter(50, 10) # *w3--batch_size x 10
        self.b3 = nn.Parameter(1, 10)
        

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        x1 = nn.Linear(x, self.w1)
        x1_b1 = nn.AddBias(x1, self.b1)
        relu1 = nn.ReLU(x1_b1)
        x2 = nn.Linear(relu1, self.w2)
        x2_b2 = nn.AddBias(x2, self.b2)
        relu2 = nn.ReLU(x2_b2)
        x3 = nn.Linear(relu2, self.w3)
        x3_b3 = nn.AddBias(x3, self.b3)
        return x3_b3
 
    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        time_start=time.time()
        multiplier = -0.06 # note: multiplier is different from learning rate, it must be negative
        batch_size = 15
        # update
        while dataset.get_validation_accuracy() < 0.975:
            for x, y in dataset.iterate_once(batch_size):
                loss = self.get_loss(x, y)
                grad_wrt_w1, grad_wrt_b1, grad_wrt_w2, grad_wrt_b2, grad_wrt_w3, grad_wrt_b3 = nn.gradients(loss, [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3])
                self.w1.update(grad_wrt_w1, multiplier)
                self.b1.update(grad_wrt_b1, multiplier)
                self.w2.update(grad_wrt_w2, multiplier)
                self.b2.update(grad_wrt_b2, multiplier)
                self.w3.update(grad_wrt_w3, multiplier)
                self.b3.update(grad_wrt_b3, multiplier)
        time_end=time.time()
        print('totally cost ',time_end-time_start)

        # multipler = -0.01, time = 183s
        # multipler = -0.05, time = 125
        # multipler = -0.06, time = 107s
        # multipler = -0.07, time = 124s

        # batch_size = 10， time = 107
        # batch_size = 5， time = 419


class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        # start with 1 layers
        self.w1 = nn.Parameter(47, 100)
        self.b1 = nn.Parameter(1, 100)


        # start with 1 layer hidden
        self.w2 = nn.Parameter(100, 100)
        self.b2 = nn.Parameter(1, 100)


        # output layer
        self.w3 = nn.Parameter(100, 50)
        self.b3 = nn.Parameter(1, 50)
        self.w4 = nn.Parameter(50, 30)
        self.b4 = nn.Parameter(1, 30)
        self.w5 = nn.Parameter(30, 10)
        self.b5 = nn.Parameter(1, 10)
        self.w6 = nn.Parameter(10, 5)
        self.b6 = nn.Parameter(1, 5)

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        # f_initial: treat the first letter
        x1 = nn.Linear(xs[0], self.w1)
        x1_b1 = nn.AddBias(x1, self.b1)
        h = x1_b1

        # f
        for i in range(1, len(xs)):
            # sum
            z = nn.Add(nn.AddBias(nn.Linear(xs[i], self.w1), self.b1), nn.AddBias(nn.Linear(h, self.w2), self.b2))
            #run
            x1_b1 = nn.AddBias(z, self.b1)
            h = x1_b1 # set h

        # output
        x3 = nn.Linear(h, self.w3)
        x3_b3 = nn.AddBias(x3, self.b3)
        relu1 = nn.ReLU(x3_b3)
        x4 = nn.Linear(relu1, self.w4)
        x4_b4 = nn.AddBias(x4, self.b4)
        relu2 = nn.ReLU(x4_b4)
        x5 = nn.Linear(relu2, self.w5)
        x5_b5 = nn.AddBias(x5, self.b5)
        relu3 = nn.ReLU(x5_b5)
        x6 = nn.Linear(relu3, self.w6)
        x6_b6 = nn.AddBias(x6, self.b6)
        return x6_b6

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(xs), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        time_start=time.time()
        multiplier = -0.01 # note: multiplier is different from learning rate, it must be negative
        batch_size = 10
        # update
        while dataset.get_validation_accuracy() < 0.85:
            for x, y in dataset.iterate_once(batch_size):
                loss = self.get_loss(x, y)
                grad_wrt_w1, grad_wrt_b1, grad_wrt_w2, grad_wrt_b2, grad_wrt_w3, grad_wrt_b3, grad_wrt_w4, grad_wrt_b4, grad_wrt_w5, grad_wrt_b5, grad_wrt_w6, grad_wrt_b6  = nn.gradients(loss, [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.w4, self.b4, self.w5, self.b5, self.w6, self.b6])
                self.w1.update(grad_wrt_w1, multiplier)
                self.b1.update(grad_wrt_b1, multiplier)
                self.w2.update(grad_wrt_w2, multiplier)
                self.b2.update(grad_wrt_b2, multiplier)
                self.w3.update(grad_wrt_w3, multiplier)
                self.b3.update(grad_wrt_b3, multiplier)
                self.w4.update(grad_wrt_w4, multiplier)
                self.b4.update(grad_wrt_b4, multiplier)
                self.w5.update(grad_wrt_w5, multiplier)
                self.b5.update(grad_wrt_b5, multiplier)
                self.w6.update(grad_wrt_w6, multiplier)
                self.b6.update(grad_wrt_b6, multiplier)
        time_end=time.time()
        print('totally cost ',time_end-time_start)
