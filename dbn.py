from util import *
from rbm import RestrictedBoltzmannMachine

class DeepBeliefNet():    

    ''' 
    For more details : Hinton, Osindero, Teh (2006). A fast learning algorithm for deep belief nets. https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf

    network          : [top] <---> [pen] ---> [hid] ---> [vis] 
                               `-> [lbl] 
    lbl : label
    top : top
    pen : penultimate
    hid : hidden
    vis : visible
    '''
    
    def __init__(self, sizes, image_size, n_labels, batch_size):

        """
        Args:
          sizes: Dictionary of layer names and dimensions
          image_size: Image dimension of data
          n_labels: Number of label categories
          batch_size: Size of mini-batch
        """

        self.rbm_stack = {
            
            'vis--hid' : RestrictedBoltzmannMachine(ndim_visible=sizes["vis"], ndim_hidden=sizes["hid"],
                                                    is_bottom=True, image_size=image_size, batch_size=batch_size),
            
            'hid--pen' : RestrictedBoltzmannMachine(ndim_visible=sizes["hid"], ndim_hidden=sizes["pen"], batch_size=batch_size),
            
            'pen+lbl--top' : RestrictedBoltzmannMachine(ndim_visible=sizes["pen"]+sizes["lbl"], ndim_hidden=sizes["top"],
                                                        is_top=True, n_labels=n_labels, batch_size=batch_size)
        }
        
        self.sizes = sizes

        self.image_size = image_size

        self.batch_size = batch_size
        
        self.n_gibbs_recog = 30
        
        self.n_gibbs_gener = 200
        
        self.n_gibbs_wakesleep = 5

        self.print_period = 2000
        
        return

    def recognize(self,true_img,true_lbl,output_dir=None):

        """Recognize/Classify the data into label categories and calculate the accuracy

        Args:
          true_imgs: visible data shaped (number of samples, size of visible layer)
          true_lbl: true labels shaped (number of samples, size of label layer). Used only for calculating accuracy, not driving the net
        """
        
        n_samples = true_img.shape[0]
        predicted_lbl = np.zeros((n_samples, self.sizes["lbl"]))
        label_history = np.zeros(
            (self.n_gibbs_recog, n_samples, self.sizes["lbl"])
        )
        bottom = self.rbm_stack["vis--hid"]
        middle = self.rbm_stack["hid--pen"]
        top = self.rbm_stack["pen+lbl--top"]

        # Batch processing to avoid memory issues
        for start in range(0, n_samples, self.batch_size):
            end = min(start + self.batch_size, n_samples)
            vis = true_img[start:end] # visible layer gets the image data

            _, hid = bottom.get_h_given_v_dir(vis)
            _, pen = middle.get_h_given_v_dir(hid)

            # Initially, all labels are set to uniform distribution
            lbl = np.full((len(vis), self.sizes["lbl"]), 1.0 / self.sizes["lbl"]) # start the net by telling you know nothing about labels        

            top_visible = np.concatenate((pen, lbl), axis=1)
        # [TODO TASK 4.2] fix the image data in the visible layer and drive the network bottom to top. In the top RBM, run alternating Gibbs sampling \
        # and read out the labels (replace pass below and 'predicted_lbl' to your predicted labels).
        # NOTE : inferring entire train/test set may require too much compute memory (depends on your system). In that case, divide into mini-batches.
        
            for step in range(self.n_gibbs_recog):
                _, top_hidden = top.get_h_given_v(top_visible)
                visible_probs, top_visible = top.get_v_given_h(top_hidden)

                top_visible[:, :self.sizes["pen"]] = pen  # Fix the penultimate layer to the values obtained from the bottom-up pass
                label_history[step, start:end] = (
                    visible_probs[:, -self.sizes["lbl"]:]
                )

            predicted_lbl[start:end] = visible_probs[:, -self.sizes["lbl"]:]

        true_classes = np.argmax(true_lbl, axis=1)

        accuracy_by_step = np.mean(
            np.argmax(label_history, axis=2) == true_classes[None, :],
            axis=1
        )

        self.label_history = label_history
        self.accuracy_by_step = accuracy_by_step

        if output_dir is not None:
            from pathlib import Path
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            steps = np.arange(1, self.n_gibbs_recog + 1)
            np.savetxt(out / "accuracy_by_step.csv",
                       np.column_stack((steps, accuracy_by_step)), delimiter=",",
                       header="gibbs_step,accuracy_fraction", comments="")
            np.save(out / "label_history.npy", label_history)
            np.save(out / "true_classes.npy", true_classes)
            fig, ax = plt.subplots()
            ax.plot(steps, 100 * accuracy_by_step, marker="o")
            ax.set_xlabel("Gibbs step")
            ax.set_ylabel("Accuracy (%)")
            ax.set_title("Recognition on %d images" % n_samples)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(out / "accuracy_by_step.png")
            plt.close(fig)

            final_classes = np.argmax(predicted_lbl, axis=1)
            for category, mask in (
                ("correct", final_classes == true_classes),
                ("incorrect", final_classes != true_classes)
            ):
                candidates = np.flatnonzero(mask)
                if len(candidates) == 0:
                    continue
                index = candidates[0]
                # Include the uniform initial probabilities as step zero.
                probabilities = np.vstack((
                    np.full((1, self.sizes["lbl"]), 1.0 / self.sizes["lbl"]),
                    label_history[:, index, :]
                ))
                np.savetxt(out / (category + "_label_probabilities.csv"),
                           np.column_stack((np.arange(len(steps) + 1), probabilities)),
                           delimiter=",", comments="",
                           header="gibbs_step," + ",".join(
                               "label_%d" % digit for digit in range(self.sizes["lbl"])))
                fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                axes[0].imshow(true_img[index].reshape(self.image_size),
                               cmap="gray", vmin=0, vmax=1)
                axes[0].set_title("Image %d: true=%d, predicted=%d" %
                                  (index, true_classes[index], final_classes[index]))
                axes[0].axis("off")
                for digit in range(self.sizes["lbl"]):
                    axes[1].plot(np.arange(len(steps) + 1), probabilities[:, digit],
                                 label=str(digit))
                axes[1].set_xlabel("Gibbs step")
                axes[1].set_ylabel("Label probability")
                axes[1].set_ylim(0, 1)
                axes[1].legend(title="Digit", ncol=2)
                fig.tight_layout()
                fig.savefig(out / (category + "_label_convergence.png"))
                plt.close(fig)
            (out / "evaluation_summary.txt").write_text(
                "samples=%d\ngibbs_steps=%d\nfinal_accuracy=%.6f\n"
                "Accuracy is a fraction; probabilities are from one stochastic run.\n" %
                (n_samples, self.n_gibbs_recog, accuracy_by_step[-1]))

        accuracy = np.mean(np.argmax(predicted_lbl, axis=1) == np.argmax(true_lbl, axis=1)) 
            
        print("accuracy = %.2f%%" % (100.0 * accuracy))
        
        return predicted_lbl

    def generate(self,true_lbl,name):
        
        """Generate data from labels

        Args:
          true_lbl: true labels shaped (number of samples, size of label layer)
          name: string used for saving a video of generated visible activations
        """
        
        n_sample = true_lbl.shape[0]
        
        records = []        
        fig,ax = plt.subplots(1,1,figsize=(3,3))
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        ax.set_xticks([]); ax.set_yticks([])

        lbl = true_lbl

        bottom = self.rbm_stack["vis--hid"]
        middle = self.rbm_stack["hid--pen"]
        top = self.rbm_stack["pen+lbl--top"]

        pen = sample_binary(np.full((n_sample, self.sizes["pen"]), 0.5))

        top_visible = np.concatenate((pen, true_lbl), axis=1)

        # [Completed TASK 4.2] fix the label in the label layer and run alternating Gibbs sampling in the top RBM. From the top RBM, drive the network \ 
        # top to the bottom visible layer (replace 'vis' from random to your generated visible layer).
            
        for _ in range(self.n_gibbs_gener):

            _, top_hidden = top.get_h_given_v(top_visible)
            _, top_visible = top.get_v_given_h(top_hidden)

            top_visible[:, -self.sizes["lbl"]:] = true_lbl  # Fix the label layer to the true label

            pen = top_visible[:, :self.sizes["pen"]]
            _, hid = middle.get_v_given_h_dir(pen)
            vis, _ = bottom.get_v_given_h_dir(hid)
            
            records.append( [ ax.imshow(vis.reshape(self.image_size), cmap="bwr", vmin=0, vmax=1, animated=True, interpolation=None) ] )
            
        anim = stitch_video(fig,records).save("%s.generate%d.mp4"%(name,np.argmax(true_lbl)))            
            
        return

    def train_greedylayerwise(self, vis_trainset, lbl_trainset, n_epochs=10):

        """
        Greedy layer-wise training by stacking RBMs. This method first tries to load previous saved parameters of the entire RBM stack. 
        If not found, learns layer-by-layer (which needs to be completed) .
        Notice that once you stack more layers on top of a RBM, the weights are permanently untwined.

        Args:
          vis_trainset: visible data shaped (size of training set, size of visible layer)
            lbl_trainset: label data shaped (size of training set, size of label layer)
            n_epochs: number of epochs of learning (each epoch learns the entire training set)
        """

        try :

            self.loadfromfile_rbm(loc="trained_rbm",name="vis--hid")
            self.rbm_stack["vis--hid"].untwine_weights()            
            
            self.loadfromfile_rbm(loc="trained_rbm",name="hid--pen")
            self.rbm_stack["hid--pen"].untwine_weights()
            
            self.loadfromfile_rbm(loc="trained_rbm",name="pen+lbl--top")        

        except IOError :

            # [TODO TASK 4.2] use CD-1 to train all RBMs greedily
        
            print ("training vis--hid")

            bottom = self.rbm_stack["vis--hid"]
            bottom.cd1(visible_trainset=vis_trainset, n_epochs=n_epochs,
                       output_dir="report_images/dbn_pretraining_weight_decay_20epochs/vis_hid")

            self.savetofile_rbm(loc="trained_rbm",name="vis--hid")
            # Higher RBMs receive probabilistic representations during pretraining.
            hid_trainset, _ = bottom.get_h_given_v(vis_trainset)

            print ("training hid--pen")

            middle = self.rbm_stack["hid--pen"]
            middle.cd1(
                visible_trainset=hid_trainset, n_epochs=n_epochs,
                output_dir="report_images/dbn_pretraining_weight_decay_20epochs/hid_pen"
            )

            self.savetofile_rbm(loc="trained_rbm",name="hid--pen")

            pen_trainset, _ = middle.get_h_given_v(hid_trainset)
        
            self.rbm_stack["vis--hid"].untwine_weights()                       

            print ("training pen+lbl--top")

            top_trainset = np.concatenate((pen_trainset, lbl_trainset), axis=1)

            top = self.rbm_stack["pen+lbl--top"]
            top.cd1(visible_trainset=top_trainset, n_epochs=n_epochs,
                    output_dir="report_images/dbn_pretraining_weight_decay_20epochs/pen_lbl_top")
            """ 
            CD-1 training for pen+lbl--top 
            """
            self.rbm_stack["hid--pen"].untwine_weights()
            self.savetofile_rbm(loc="trained_rbm",name="pen+lbl--top")            

        return    

    def train_wakesleep_finetune(self, vis_trainset, lbl_trainset, n_iterations):

        """
        Wake-sleep method for learning all the parameters of network. 
        First tries to load previous saved parameters of the entire network.

        Args:
          vis_trainset: visible data shaped (size of training set, size of visible layer)
          lbl_trainset: label data shaped (size of training set, size of label layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """
        
        print ("\ntraining wake-sleep..")

        try :
            
            self.loadfromfile_dbn(loc="trained_dbn",name="vis--hid")
            self.loadfromfile_dbn(loc="trained_dbn",name="hid--pen")
            self.loadfromfile_rbm(loc="trained_dbn",name="pen+lbl--top")
            
        except IOError :            

            self.n_samples = vis_trainset.shape[0]

            for it in range(n_iterations):            
                                                
                # [TODO TASK 4.3] wake-phase : drive the network bottom to top using fixing the visible and label data.

                # [TODO TASK 4.3] alternating Gibbs sampling in the top RBM for k='n_gibbs_wakesleep' steps, also store neccessary information for learning this RBM.

                # [TODO TASK 4.3] sleep phase : from the activities in the top RBM, drive the network top to bottom.

                # [TODO TASK 4.3] compute predictions : compute generative predictions from wake-phase activations, and recognize predictions from sleep-phase activations.
                # Note that these predictions will not alter the network activations, we use them only to learn the directed connections.
                
                # [TODO TASK 4.3] update generative parameters : here you will only use 'update_generate_params' method from rbm class.

                # [TODO TASK 4.3] update parameters of top rbm : here you will only use 'update_params' method from rbm class.

                # [TODO TASK 4.3] update generative parameters : here you will only use 'update_recognize_params' method from rbm class.

                if it % self.print_period == 0 : print ("iteration=%7d"%it)
                        
            self.savetofile_dbn(loc="trained_dbn",name="vis--hid")
            self.savetofile_dbn(loc="trained_dbn",name="hid--pen")
            self.savetofile_rbm(loc="trained_dbn",name="pen+lbl--top")            

        return

    
    def loadfromfile_rbm(self,loc,name):
        
        self.rbm_stack[name].weight_vh = np.load("%s/rbm.%s.weight_vh.npy"%(loc,name))
        self.rbm_stack[name].bias_v    = np.load("%s/rbm.%s.bias_v.npy"%(loc,name))
        self.rbm_stack[name].bias_h    = np.load("%s/rbm.%s.bias_h.npy"%(loc,name))
        print ("loaded rbm[%s] from %s"%(name,loc))
        return
        
    def savetofile_rbm(self,loc,name):
        
        np.save("%s/rbm.%s.weight_vh"%(loc,name), self.rbm_stack[name].weight_vh)
        np.save("%s/rbm.%s.bias_v"%(loc,name),    self.rbm_stack[name].bias_v)
        np.save("%s/rbm.%s.bias_h"%(loc,name),    self.rbm_stack[name].bias_h)
        return
    
    def loadfromfile_dbn(self,loc,name):
        
        self.rbm_stack[name].weight_v_to_h = np.load("%s/dbn.%s.weight_v_to_h.npy"%(loc,name))
        self.rbm_stack[name].weight_h_to_v = np.load("%s/dbn.%s.weight_h_to_v.npy"%(loc,name))
        self.rbm_stack[name].bias_v        = np.load("%s/dbn.%s.bias_v.npy"%(loc,name))
        self.rbm_stack[name].bias_h        = np.load("%s/dbn.%s.bias_h.npy"%(loc,name))
        print ("loaded rbm[%s] from %s"%(name,loc))
        return
        
    def savetofile_dbn(self,loc,name):
        
        np.save("%s/dbn.%s.weight_v_to_h"%(loc,name), self.rbm_stack[name].weight_v_to_h)
        np.save("%s/dbn.%s.weight_h_to_v"%(loc,name), self.rbm_stack[name].weight_h_to_v)
        np.save("%s/dbn.%s.bias_v"%(loc,name),        self.rbm_stack[name].bias_v)
        np.save("%s/dbn.%s.bias_h"%(loc,name),        self.rbm_stack[name].bias_h)
        return
    
