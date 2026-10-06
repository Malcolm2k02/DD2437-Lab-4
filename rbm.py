from util import *

class RestrictedBoltzmannMachine():
    '''
    For more details : A Practical Guide to Training Restricted Boltzmann Machines https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
    '''
    def __init__(self, ndim_visible, ndim_hidden, is_bottom=False, image_size=[28,28], is_top=False, n_labels=10, batch_size=10):

        """
        Args:
          ndim_visible: Number of units in visible layer.
          ndim_hidden: Number of units in hidden layer.
          is_bottom: True only if this rbm is at the bottom of the stack in a deep belief net. Used to interpret visible layer as image data with dimensions "image_size".
          image_size: Image dimension for visible layer.
          is_top: True only if this rbm is at the top of stack in deep beleif net. Used to interpret visible layer as concatenated with "n_label" unit of label data at the end. 
          n_label: Number of label categories.
          batch_size: Size of mini-batch.
        """
       
        self.ndim_visible = ndim_visible

        self.ndim_hidden = ndim_hidden

        self.is_bottom = is_bottom

        if is_bottom : self.image_size = image_size
        
        self.is_top = is_top

        if is_top : self.n_labels = 10

        self.batch_size = batch_size        
                
        self.delta_bias_v = 0

        self.delta_weight_vh = 0

        self.delta_bias_h = 0

        self.bias_v = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_visible))

        self.weight_vh = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_visible,self.ndim_hidden))

        self.bias_h = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_hidden))
        
        self.delta_weight_v_to_h = 0

        self.delta_weight_h_to_v = 0        
        
        self.weight_v_to_h = None
        
        self.weight_h_to_v = None

        self.learning_rate = 0.01
        
        self.momentum = 0.5
        self.momentum_final = 0.9
        self.momentum_warmup_epochs = 5

        self.print_period = 1000
        
        self.rf = { # receptive-fields. Only applicable when visible layer is input data
            "period" : 5000, # iteration period to visualize
            "grid" : [5,5], # size of the grid
            "ids" : np.random.randint(0,self.ndim_hidden,25) # pick some random hidden units
            }
        
        return

        
    def cd1(self, visible_trainset, n_epochs=10, output_dir=None,
            reconstruction_images=None):
        """Train with CD-1; optionally save epoch MSE, timing and figures.

        Epoch MSE averages minibatch errors measured during learning.
        reconstruction_images selects fixed images for the final comparison.
        """
        from pathlib import Path
        from time import perf_counter

        n_samples = visible_trainset.shape[0]
        if n_samples == 0 or n_epochs < 1 or self.batch_size < 1:
            raise ValueError("Training data, epochs and batch size must be positive")
        out = Path(output_dir) if output_dir is not None else None
        if out is not None:
            out.mkdir(parents=True, exist_ok=True)
        self.training_history = []
        training_start = perf_counter()
        it = 0
        print("learning CD1")

        for epoch in range(n_epochs):
            self.momentum = 0.5 if epoch < self.momentum_warmup_epochs else self.momentum_final
            print("epoch=%3d momentum=%.1f" % (epoch + 1, self.momentum))
            shuffled_indices = np.random.permutation(n_samples)
            epoch_loss_sum = 0.0
            epoch_start = perf_counter()

            for start in range(0, n_samples, self.batch_size):
                indices = shuffled_indices[start:start + self.batch_size]
                v_0 = visible_trainset[indices]
                p_h_0, h_0 = self.get_h_given_v(v_0)
                p_v_1, v_1 = self.get_v_given_h(h_0)
                p_h_1, _ = self.get_h_given_v(v_1)
                self.update_params(v_0, p_h_0, v_1, p_h_1)
                batch_loss = np.mean((v_0 - p_v_1) ** 2)
                epoch_loss_sum += batch_loss * v_0.shape[0]
                if it % self.print_period == 0:
                    print("iteration=%7d recon_loss=%4.4f" % (it, batch_loss))
                it += 1

            epoch_loss = epoch_loss_sum / n_samples
            epoch_seconds = perf_counter() - epoch_start
            self.training_history.append((epoch + 1, epoch_loss, self.momentum, epoch_seconds))
            print("epoch=%3d recon_loss=%.4f time=%.1fs" %
                  (epoch + 1, epoch_loss, epoch_seconds))
            if out is not None:
                np.savetxt(out / "epoch_metrics.csv", self.training_history,
                           delimiter=",", header="epoch,training_reconstruction_mse,momentum,seconds",
                           comments="")

        self.training_seconds = perf_counter() - training_start
        if out is not None:
            (out / "training_summary.txt").write_text(
                "hidden_units=%d\ntraining_samples=%d\nbatch_size=%d\nepochs=%d\n"
                "learning_rate=%g\nmomentum=0.5 for first %d epochs, then %g\n"
                "training_seconds=%.3f\n" %
                (self.ndim_hidden, n_samples, self.batch_size, n_epochs,
                 self.learning_rate, self.momentum_warmup_epochs,
                 self.momentum_final, self.training_seconds))
            fig, ax = plt.subplots()
            history = np.asarray(self.training_history)
            ax.plot(history[:, 0], history[:, 1], marker="o")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Training reconstruction MSE")
            ax.set_title("RBM: %d hidden units" % self.ndim_hidden)
            fig.tight_layout()
            fig.savefig(out / "reconstruction_loss.png")
            plt.close(fig)

        if self.is_bottom:
            viz_rf(weights=self.weight_vh[:, self.rf["ids"]].reshape(
                (self.image_size[0], self.image_size[1], -1)),
                it=it, grid=self.rf["grid"],
                filename=out / "receptive_fields_final.png" if out is not None else None)
            originals = (visible_trainset[:10] if reconstruction_images is None
                         else reconstruction_images[:10])
            if len(originals) == 0:
                raise ValueError("At least one reconstruction image is required")
            _, hidden = self.get_h_given_v(originals)
            reconstructions, _ = self.get_v_given_h(hidden)
            fig, axes = plt.subplots(2, len(originals),
                                     figsize=(2 * len(originals), 4), squeeze=False)
            for i in range(len(originals)):
                axes[0, i].imshow(originals[i].reshape(self.image_size),
                                  cmap="gray", vmin=0, vmax=1)
                axes[1, i].imshow(reconstructions[i].reshape(self.image_size),
                                  cmap="gray", vmin=0, vmax=1)
                axes[0, i].set_title("Original")
                axes[1, i].set_title("Reconstruction")
                axes[0, i].axis("off")
                axes[1, i].axis("off")
            fig.tight_layout()
            fig.savefig(out / "reconstructions_final.png" if out is not None
                        else "reconstruction.final.png")
            plt.close(fig)
        return

    def update_params(self,v_0,h_0,v_k,h_k):

        """Update the weight and bias parameters.

        You could also add weight decay and momentum for weight updates.

        Args:
           v_0: activities or probabilities of visible layer (data to the rbm)
           h_0: activities or probabilities of hidden layer
           v_k: activities or probabilities of visible layer
           h_k: activities or probabilities of hidden layer
           all args have shape (size of mini-batch, size of respective layer)
        """

        # [Completed TASK 4.1] get the gradients from the arguments (replace the 0s below) and update the weight and bias parameters
        
        # Retain a fraction of the previous update to smooth learning.
        self.delta_bias_v = (
            self.momentum * self.delta_bias_v
            + self.learning_rate * np.mean(v_0 - v_k, axis=0)
        )
        self.delta_weight_vh = (
            self.momentum * self.delta_weight_vh
            + self.learning_rate * (v_0.T @ h_0 - v_k.T @ h_k) / v_0.shape[0]
        )
        self.delta_bias_h = (
            self.momentum * self.delta_bias_h
            + self.learning_rate * np.mean(h_0 - h_k, axis=0)
        )
        
        self.bias_v += self.delta_bias_v
        self.weight_vh += self.delta_weight_vh
        self.bias_h += self.delta_bias_h
        
        return

    def get_h_given_v(self,visible_minibatch):
        
        """Compute probabilities p(h|v) and activations h ~ p(h|v) 

        Uses undirected weight "weight_vh" and bias "bias_h"
        
        Args: 
           visible_minibatch: shape is (size of mini-batch, size of visible layer)
        Returns:        
           tuple ( p(h|v) , h) 
           both are shaped (size of mini-batch, size of hidden layer)
        """
        
        assert self.weight_vh is not None

        n_samples = visible_minibatch.shape[0]

        # [Completed TASK 4.1] compute probabilities and activations (samples from probabilities) of hidden layer (replace the zeros below) 
        support = visible_minibatch @ self.weight_vh + self.bias_h
        probabilities = sigmoid(support)
        activations = sample_binary(probabilities)
        
        return probabilities, activations


    def get_v_given_h(self,hidden_minibatch):
        
        """Compute probabilities p(v|h) and activations v ~ p(v|h)

        Uses undirected weight "weight_vh" and bias "bias_v"
        
        Args: 
           hidden_minibatch: shape is (size of mini-batch, size of hidden layer)
        Returns:        
           tuple ( p(v|h) , v) 
           both are shaped (size of mini-batch, size of visible layer)
        """
        
        assert self.weight_vh is not None

        n_samples = hidden_minibatch.shape[0]

        if self.is_top:

            """
            Here visible layer has both data and labels. Compute total input for each unit (identical for both cases), \ 
            and split into two parts, something like support[:, :-self.n_labels] and support[:, -self.n_labels:]. \
            Then, for both parts, use the appropriate activation function to get probabilities and a sampling method \
            to get activities. The probabilities as well as activities can then be concatenated back into a normal visible layer.
            """

            # [Completed TASK 4.1] compute probabilities and activations (samples from probabilities) of visible layer (replace the pass below). \
            # Note that this section can also be postponed until TASK 4.2, since in this task, stand-alone RBMs do not contain labels in visible layer.
            support = hidden_minibatch @ self.weight_vh.T + self.bias_v
            data_support = support[:, :-self.n_labels]
            label_support = support[:, -self.n_labels:]

            data_probabilities = sigmoid(data_support)
            label_probabilities = softmax(label_support)

            data_activations = sample_binary(data_probabilities)
            label_activations = sample_categorical(label_probabilities)

            probabilities = np.concatenate((data_probabilities, label_probabilities), axis=1)
            activations = np.concatenate((data_activations, label_activations), axis=1)
            
            
        else:
                        
            # [Completed TASK 4.1] compute probabilities and activations (samples from probabilities) of visible layer (replace the pass and zeros below)             
            support = hidden_minibatch @ self.weight_vh.T + self.bias_v
            probabilities = sigmoid(support)
            activations = sample_binary(probabilities)
        
        return probabilities, activations


    
    """ rbm as a belief layer : the functions below do not have to be changed until running a deep belief net """

    

    def untwine_weights(self):
        
        self.weight_v_to_h = np.copy( self.weight_vh )
        self.weight_h_to_v = np.copy( np.transpose(self.weight_vh) )
        self.weight_vh = None

    def get_h_given_v_dir(self,visible_minibatch):

        """Compute probabilities p(h|v) and activations h ~ p(h|v)

        Uses directed weight "weight_v_to_h" and bias "bias_h"
        
        Args: 
           visible_minibatch: shape is (size of mini-batch, size of visible layer)
        Returns:        
           tuple ( p(h|v) , h) 
           both are shaped (size of mini-batch, size of hidden layer)
        """
        
        assert self.weight_v_to_h is not None

        n_samples = visible_minibatch.shape[0]

        # [Completed TASK 4.2] perform same computation as the function 'get_h_given_v' but with directed connections (replace the zeros below) 
        support = visible_minibatch @ self.weight_v_to_h + self.bias_h
        probabilities = sigmoid(support)
        activations = sample_binary(probabilities)
        return probabilities, activations


    def get_v_given_h_dir(self,hidden_minibatch):


        """Compute probabilities p(v|h) and activations v ~ p(v|h)

        Uses directed weight "weight_h_to_v" and bias "bias_v"
        
        Args: 
           hidden_minibatch: shape is (size of mini-batch, size of hidden layer)
        Returns:        
           tuple ( p(v|h) , v) 
           both are shaped (size of mini-batch, size of visible layer)
        """
        
        assert self.weight_h_to_v is not None
        
        n_samples = hidden_minibatch.shape[0]
        
        if self.is_top:

            """
            Here visible layer has both data and labels. Compute total input for each unit (identical for both cases), \ 
            and split into two parts, something like support[:, :-self.n_labels] and support[:, -self.n_labels:]. \
            Then, for both parts, use the appropriate activation function to get probabilities and a sampling method \
            to get activities. The probabilities as well as activities can then be concatenated back into a normal visible layer.
            """
            
            # [Completed TASK 4.2] Note that even though this function performs same computation as 'get_v_given_h' but with directed connections,
            # this case should never be executed : when the RBM is a part of a DBN and is at the top, it will have not have directed connections.
            # Appropriate code here is to raise an error (replace pass below)
            
            raise ValueError("The top RBM must use undirected connections")
            
        else:
                        
            # [Completed TASK 4.2] performs same computaton as the function 'get_v_given_h' but with directed connections (replace the pass and zeros below)             
            support = hidden_minibatch @ self.weight_h_to_v + self.bias_v
            probabilities = sigmoid(support)
            activations = sample_binary(probabilities)
            
            
        return probabilities, activations        
        
    def update_generate_params(self,inps,trgs,preds):
        
        """Update generative weight "weight_h_to_v" and bias "bias_v"
        
        Args:
           inps: activities or probabilities of input unit
           trgs: activities or probabilities of output unit (target)
           preds: activities or probabilities of output unit (prediction)
           all args have shape (size of mini-batch, size of respective layer)
        """

        # [TODO TASK 4.3] find the gradients from the arguments (replace the 0s below) and update the weight and bias parameters.
        
        self.delta_weight_h_to_v += 0
        self.delta_bias_v += 0
        
        self.weight_h_to_v += self.delta_weight_h_to_v
        self.bias_v += self.delta_bias_v 
        
        return
    
    def update_recognize_params(self,inps,trgs,preds):
        
        """Update recognition weight "weight_v_to_h" and bias "bias_h"
        
        Args:
           inps: activities or probabilities of input unit
           trgs: activities or probabilities of output unit (target)
           preds: activities or probabilities of output unit (prediction)
           all args have shape (size of mini-batch, size of respective layer)
        """

        # [TODO TASK 4.3] find the gradients from the arguments (replace the 0s below) and update the weight and bias parameters.

        self.delta_weight_v_to_h += 0
        self.delta_bias_h += 0

        self.weight_v_to_h += self.delta_weight_v_to_h
        self.bias_h += self.delta_bias_h
        
        return    
