clear all; clc; clf;
close all; %% close all figure windows
data = importdata('Iris_train.txt');
seq = randperm(length(data)); %% random sequence from 1 to length of data
data = data(seq,:); %% random sort the data
feature = data(:,1:4);
target = data(:,5);
neurons = 48;

%% normalization
for i = 1:4
    feature(:,i) = (feature(:,i)-min(feature(:,i)))/(max(feature(:,i))-min(feature(:,i))); 
end

%% input value = gaussian function x-axis index, thus convert decimal to integer
input = feature*10000; % Match the index of Gaussian receptive field matrix 
input = fix(input);  % exclude the value originally below 4th decimal point

% Avoid the "1" value to be gaussian matrix index(should be positive integer)
for i = 1:size(input,1)
    for j = 1:size(input,2)
        if input(i,j) == 0
            input(i,j) = 1;
        end
    end
end

%% Encoding
% Single input variable encoded by 12 GRF neurons
InputSpike = zeros(length(input),neurons); % Training samples, 48 activate values(spike times)
for i = 1:length(input)
  InputSpike(i,:) = GRF(input(i,:));   % input each variables to Gaussian receptive field
end

% Ignore the activate value < 0.1 which is not able to simulate a spike
for i = 1:length(InputSpike)
    for j = 1:4
        if InputSpike(i,j) < 0.1
            InputSpike(i,j) = 0;
        end
    end
end

InputSpike = 100*(1 - InputSpike); %% linear transform to spike time: t = 1 for train
                                   % = 0, t = 100 for the maximum value
InputSpike = round(InputSpike); % spike times with 1 ms precision
                                                  
for i = 1:size(InputSpike,1)  % Adjust t = 0 firing to t = 1
    for j = 1:size(InputSpike,2)
        if InputSpike(i,j) == 0
            InputSpike(i,j) = 1;
        end
    end
end

%% take 120 for train data, 30 for test data
test = InputSpike(101:150,:);
test_target = target(101:150);
train_target = target(1:100);
InputSpike = InputSpike(1:100,:); %% set 'train' = 'InputSpike'

%% Temporal learning
iteration = 50;
V = zeros(3,100);
V_rest = 0;
weight = rand(neurons,3);
threshold = 1;  % set up V threshold: try and error
T = 100; %% time encoding window
firing_time = zeros(1,neurons); %% time-to-first-spike for 48 neurons
Response = zeros(1,neurons); %% K(t) response value array
lr = 0.005;
Tou_m = 1.5;
Time = 1:1:T;
correct_rate = zeros(1,iteration);

for Iterate = 1:iteration  %% Run 100 times
   
      %% Test the correct rate each time
   correct = 0;
   for test_sample = 1:size(test,1)
       t_max = 100*ones(1,3);  %% default: all neuron reach max V at the end
       Max_state = zeros(1,3); % strongest state of each output neuron
       % In each iteration, T = 100ms
        for t = 1:T                
            for neuron = 1:neurons %% Response function for 48 neurons at time t
                     Response(neuron) = K(t,test(test_sample,neuron));              
            end
            % Calculate PSP
            for j = 1:3                
               V(j,t) = Response*weight(:,j) + V_rest;            
            end           
        end         
        %% find t_max: first index that V cross threshold
        for j = 1:3
            for timing = 1:T
                if V(j,timing) >= threshold
                    t_max(j) = timing;
                    Max_state(j) = V(j,timing);
                    break; % first time V > threshold
                end
            end     
           
           if t_max(j) < T   % neuron j has fired
              V(j,t_max(j):end) = V(j,t_max(j)).*exp(-(Time(t_max(j):end)-Time(t_max(j)))/Tou_m);
           end
        end
        
        [~,output_class] = min(t_max); 
        if test_target(test_sample) == output_class
              correct = correct + 1;
        end
          
   end
   
   correct_rate(Iterate) = correct/size(test,1);
   
   %% Training
   % Train for class 1
    for samples = 1:size(InputSpike,1)  %% Training samples for each iteration 
        t_max = 100*ones(1,3);  %% default: all neuron reach max V at the end
        Max_state = zeros(1,3); % strongest state of each output neuron
        % In each iteration, T = 100ms
        for t = 1:T                  
            for neuron = 1:neurons %% Response function for 48 neurons at time t
                Response(neuron) = K(t,InputSpike(samples,neuron));              
            end       
            % Calculate PSP
            for j = 1:3                
               V(j,t) = Response*weight(:,j) + V_rest;            
            end
        end           
        %% find t_max: first index that V cross threshold
        for j = 1:3
            for timing = 1:T
                if V(j,timing) >= threshold
                    t_max(j) = timing;
                    Max_state(j) = V(j,timing);
                    break;
                end
            end        
           
           if t_max(j) < T
              V(j,t_max(j):end) = V(j,t_max(j)).*exp(-(Time(t_max(j):end)-Time(t_max(j)))/Tou_m);
           end
        end
        [~,output_class] = min(t_max);
        %% weight modify when error occurs       
        if train_target(samples) == 1   
           if output_class ~= 1
             for j = 1:3               
                if j == 1 %% error in target neuron
                    if Max_state(j) < threshold %% if P+ error occurs
                        for i = 1:neurons
                            %% for all t_i < t_max
                            if InputSpike(samples,i) < t_max(j) 
                                %% weight modified
                                weight(i,j) = weight(i,j) + ...
                                    lr*K(t_max(j),InputSpike(samples,i));
                            end
                        end
                    end
                elseif j ~= 1 %% error on other 2 output neurons  
                   if Max_state(j) >= threshold %% if P- error occurs
                       for i = 1:neurons
                            %% for all t_i < t_max
                            if InputSpike(samples,i) < t_max(j) 
                                %% weight modified
                                weight(i,j) = weight(i,j) - ...
                                    lr*K(t_max(j),InputSpike(samples,i));
                            end
                       end
                   end
                end  
             end
           elseif output_class == 1  
               for j = 1:3
                   if j ~= 1
                       if Max_state(j) >= threshold
                           for i = 1:neurons
                               if InputSpike(samples,i) < t_max(j)
                                   weight(i,j) = weight(i,j) - ...
                                     lr*K(t_max(j),InputSpike(samples,i));
                               end
                           end
                       end
                   end
               end
           end     
        %% for neurons that fired but weaker than target neuron     
        elseif train_target(samples) ~= 1
               if Max_state(1) >= threshold
                    for i = 1:neurons %% P- error occurs
                         %% for all t_i < t_max
                         if InputSpike(samples,i) < t_max(1) 
                             %% weight modified
                             weight(i,1) = weight(i,1) - ...
                                 lr*K(t_max(1),InputSpike(samples,i));
                         end 
                    end
               end
        end         
        
    end    
   
    % Train for class 2
    for samples = 1:size(InputSpike,1)  %% Training samples for each iteration 
        t_max = 100*ones(1,3);  %% default: all neuron reach max V at the end
        Max_state = zeros(1,3); % strongest state of each output neuron
        % In each iteration, T = 100ms
        for t = 1:T                  
            for neuron = 1:neurons %% Response function for 48 neurons at time t
                Response(neuron) = K(t,InputSpike(samples,neuron));              
            end       
            % Calculate PSP
            for j = 1:3                
               V(j,t) = Response*weight(:,j) + V_rest;            
            end
        end           
        %% find t_max: first index that V cross threshold
        for j = 1:3
            for timing = 1:T
                if V(j,timing) >= threshold
                    t_max(j) = timing;
                    Max_state(j) = V(j,timing);
                    break;
                end
            end       
            
           if t_max(j) < T 
              V(j,t_max(j):end) = V(j,t_max(j)).*exp(-(Time(t_max(j):end)-Time(t_max(j)))/Tou_m);
           end
        end
        
        [~,output_class] = min(t_max);
        %% weight modify when error occurs       
        if train_target(samples) == 2   
           if output_class ~= 2  % [1.5 1 0] or [1 0 1]...
             for j = 1:3               
                if j == 2 %% error in target neuron
                    if Max_state(j) < threshold %% if P+ error occurs
                        for i = 1:neurons
                            %% for all t_i < t_max
                            if InputSpike(samples,i) < t_max(j) 
                                %% weight modified
                                weight(i,j) = weight(i,j) + ...
                                    lr*K(t_max(j),InputSpike(samples,i));
                            end
                        end
                    end
                elseif j ~= 2 %% error on other 2 output neurons  
                   if Max_state(j) >= threshold %% if P- error occurs
                       for i = 1:neurons
                            %% for all t_i < t_max
                            if InputSpike(samples,i) < t_max(j) 
                                %% weight modified
                                weight(i,j) = weight(i,j) - ...
                                    lr*K(t_max(j),InputSpike(samples,i));
                            end
                       end
                       
                   end
                end 
             end 
           elseif output_class == 2   %% [1 1.5 1]
               for j = 1:3
                   if j ~= 2
                     if Max_state(j) >= threshold
                         for i = 1:neurons
                             if InputSpike(samples,i) < t_max(j)
                                 weight(i,j) = weight(i,j) - ...
                                     lr*K(t_max(j),InputSpike(samples,i));
                             end
                         end
                     end
                   end
               end
           end 
           
        %% for neurons that fired but weaker than target neuron     
        elseif train_target(samples) ~= 2
               if Max_state(2) >= threshold
                    for i = 1:neurons %% P- error occurs
                         %% for all t_i < t_max
                         if InputSpike(samples,i) < t_max(2) 
                             %% weight modified
                             weight(i,2) = weight(i,2) - ...
                                 lr*K(t_max(2),InputSpike(samples,i));
                         end 
                    end
               end
        end         
        
    end 
    
    % Train for class 3
    for samples = 1:size(InputSpike,1)  %% Training samples for each iteration 
        t_max = 100*ones(1,3);  %% default: all neuron reach max V at the end
        Max_state = zeros(1,3); % strongest state of each output neuron
        % In each iteration, T = 100ms
        for t = 1:T                  
            for neuron = 1:neurons %% Response function for 48 neurons at time t
                Response(neuron) = K(t,InputSpike(samples,neuron));              
            end       
            % Calculate PSP
            for j = 1:3                
               V(j,t) = Response*weight(:,j) + V_rest;            
            end
        end           
        %% find t_max: first index that V cross threshold
        for j = 1:3
            for timing = 1:T
                if V(j,timing) >= threshold
                    t_max(j) = timing;
                    Max_state(j) = V(j,timing);
                    break;
                end
            end       
            
           if t_max(j) < T 
              V(j,t_max(j):end) = V(j,t_max(j)).*exp(-(Time(t_max(j):end)-Time(t_max(j)))/Tou_m);
           end
        end
        
        [~,output_class] = min(t_max);
        %% weight modify when error occurs       
        if train_target(samples) == 3   
           if output_class ~= 3  
             for j = 1:3               
                if j == 3 %% error in target neuron
                    if Max_state(j) < threshold %% if P+ error occurs
                        for i = 1:neurons
                            if InputSpike(samples,i) < t_max(j) 
                                weight(i,j) = weight(i,j) + ...
                                    lr*K(t_max(j),InputSpike(samples,i));
                            end
                        end
                    end
                elseif j ~= 3   
                   if Max_state(j) >= threshold 
                       for i = 1:neurons
                            if InputSpike(samples,i) < t_max(j) 
                                weight(i,j) = weight(i,j) - ...
                                    lr*K(t_max(j),InputSpike(samples,i));
                            end
                       end
                   end
                end 
             end 
           elseif output_class == 3
               for j = 1:3
                   if j ~= 3
                     if Max_state(j) >= threshold
                         for i = 1:neurons
                             if InputSpike(samples,i) < t_max(j)
                                 weight(i,j) = weight(i,j) - ...
                                     lr*K(t_max(j),InputSpike(samples,i));
                             end
                         end
                     end
                   end
               end
           end 
           
        elseif train_target(samples) ~= 3
               if Max_state(3) >= threshold
                    for i = 1:neurons
                         if InputSpike(samples,i) < t_max(3) 
                             weight(i,3) = weight(i,3) - ...
                                 lr*K(t_max(3),InputSpike(samples,i));
                         end 
                    end
               end
        end 
    end
  
end

figure(3)
plot(correct_rate)
xlabel('Iteration')
ylabel('Correct rate')
train_target(size(InputSpike,1))

time = 0:1:100;
figure;
subplot(3,1,1)
hold on
plot(V(1,:))
plot(time,threshold.*ones(length(time)),'k-.')
axis([0 100 0 1.5])
xlabel('t')
ylabel('V_1')
title('PSP Potential')
hold off

subplot(3,1,2)
hold on
plot(V(2,:))
plot(time,threshold.*ones(length(time)),'k-.')
axis([0 100 0 1.5])
xlabel('t')
ylabel('V_2')
hold off

subplot(3,1,3)
hold on
plot(V(3,:))
plot(time,threshold.*ones(length(time)),'k-.')
axis([0 100 0 1.5])
xlabel('t')
ylabel('V_3')
hold off

function response = K(t,t_i)
   time = 0:1:100;
   Tou_m = 15;
   Tou_s = Tou_m/4;
   V_0 = 1/( max(exp(-(time)/Tou_m) - exp(-(time)/Tou_s)) ); % normalize factor(divided by maximum)
   response = V_0*( exp(-(t-t_i)/Tou_m) - exp(-(t-t_i)/Tou_s) )*heavyside(t,t_i);
    % response K(t - t_i) for t >= t_i
end

function h = heavyside(t,t_i)
   if (t-t_i) >= 0
       h = 1;
   elseif (t-t_i) < 0
       h = 0;
   end
end




