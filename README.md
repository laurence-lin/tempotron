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
When a neuron accept a stimulate from pre-neuron, it's store in neuron. As long as the stimulate accessed exceed a threshold of neuron, the neuron generate a spike, that passed to the post-neuron. After the spike, the neuron "sleep" for a while and it's storage value return to 0. The neuron stay silent for a period of time, within the interval don't accept any input stimulate.
