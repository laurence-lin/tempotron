function Gaussian = GRF(input)

% input = 1*4 variable matrix
GRFneurons = 12;
%% Gaussian receptive neuron
center = ones(GRFneurons,1);  %% Xmax = 1, Xmin = 0, GRF neurons = 12
width = 1/15;  
for i = 1:GRFneurons    
    center(i) = (2*i-3)/20;
end

hold on;
x = 0:0.0001:1;  %% input variable accurancy
gaussian = zeros(GRFneurons,length(x)); 
SpikeTimes = zeros(GRFneurons,4); % 48 encoding neurons for 4 input variables
%% plot Gaussian receptive field
for i = 1:GRFneurons  
  gaussian(i,:) = exp( -((x-center(i)).^2)./(2*width*width) );
  plot(gaussian(i,:));
  axis([0 10000 0 1])
  title('Gaussian receptive field')
end
hold off;

for i = 1:4
    for j = 1:GRFneurons
        SpikeTimes(j,i) = gaussian(j,input(i)); % each spike timing for single variable to GRF neurons
    end
end

num = GRFneurons*4;
spike = zeros(1,num);  % final output = 1*48 spike times matrix
for k = 1:num
    spike(k) = SpikeTimes(k);
end

Gaussian = spike;

end