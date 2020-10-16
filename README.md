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

The latest spiking occur in t_i, then the contribution of that spiking during current timestep t is K(t - t_i). Kernel function 

K(t - t_i) is calculated by:

![image]

在此式中，spiking發生後，Kernel function 隨時間的前進而變小，spiking t_i的影響力隨時間遞減。tou_m 和 tou_s為超參數。

V0為normalization function, 將kernel function值限制在 0 和 1 內。此式中 t_i 必小於 t，因為只有過去產生的spiking會對膜電位產生貢獻。

在膜電位觸發spiking後，neuron會進入一段時間的「不應期」，而potential會返回復位電壓V_rest。

Spiking neuron的輸出只有「觸發」和「不觸發」兩種。若欲使用SNN輸出多類別: 1. 使用binary編碼，在output neuron上做binary decode產生十進位的output value.















