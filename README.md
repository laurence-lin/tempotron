# tempotron
A spiking neural network module, tempotron for classification


tempotron learning rule inspired by this thesis: 
The tempotron: a neuron that learns spike timing–based decisions

And explained detailed by this:  
Learning real-world stimuli by single-spike coding and tempotron rule 

![image](https://github.com/laurence-lin/tempotron/blob/master/Iris%20classification.jpg)


Tempotron concept:

A type of spiking neural network, an implementation to simulate the work of brain neurons. Comparing to traditional neural network, SNN is more closer to the operation of human brain neurons.

Work in spiking neuron:

When a neuron accept a stimulate from pre-neuron, it's store in neuron. As long as the stimulate accessed exceed a threshold of neuron, the neuron generate a spike, that passed to the post-neuron. After the spike, the neuron "sleep" for a while and it's storage value return to rest potential(may be zero). The neuron stay silent for a period of time, within the interval don't accept any input stimulate.

A neuron accepts all the input stimulate from pre-synaptic neuron, store the stimulates in it's potential, called 模電位。

V(t) = sigma( W_i * sigma(K(t - t_i)) ) + V_rest


K: kernel function, value locate in [0, 1]. Output of K represents the contribution of spiking happend at t_i.

The latest spiking occur in t_i, then the contribution of that spiking during current timestep t is K(t - t_i)





