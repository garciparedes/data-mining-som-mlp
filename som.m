
#
# Self-organizing map
#
# Sergio Garcia Prado
# garciparedes.me
#
clear all

# Config Values
################################################################################

inputFile = dlmread('digitos.entrena.normalizados.txt');
neuronsX = 8;
neuronsY = 12;
seasons = 50;
alphaZero = 25;

# Input Normalization
################################################################################

input = inputFile;
input([2:2:size(input,1)],:) = [];
input = [input ones(size(input, 1), 1)];
input = input ./ sqrt(sum(input.^2,2));


inputLength = size(input,1);
inputDimens = size(input,2);


expectedOutput = inputFile;
expectedOutput(:,[11:size(expectedOutput,2)]) = [];
expectedOutput([1:2:size(expectedOutput,1)],:) = [];



# SOM Inicialization
################################################################################
SOM_neurons = zeros(neuronsX, neuronsY);
SOM_neurons(:) = 1 : (neuronsX * neuronsY);

SOM_weights = rand(neuronsX * neuronsY, inputDimens) - 0.5;
for i = 1 : (neuronsX * neuronsY);
    SOM_weights(i,:) = SOM_weights(i,:) ./ norm(SOM_weights(i,:));
endfor;



# SOM Unsupervised Learning
################################################################################
radius = min(floor(neuronsX /2), floor(neuronsY /2));

for t = 1:seasons;
    for e = 1:inputLength;

        # Get Distance from input to neuron
        distances = zeros(neuronsX * neuronsY,1);
        for i = 1 : (neuronsX * neuronsY);
            distances(i) = sum(input(e,:) .* SOM_weights(i, :));
        endfor;
        
        
        [M,I] = max(abs(distances(:))); 
        [xWin,yWin] = find(SOM_neurons == I);
        
        iterator = [];
        for x = (xWin - radius) : (xWin + radius);
            if (x < 1)
                x = x + neuronsX;
            elseif(x > neuronsX)
                x = x - neuronsX;
            end

            for y = (yWin - radius) : (yWin + radius);
                if (y < 1)
                    y = y + neuronsY;
                elseif(y > neuronsY)
                    y = y - neuronsY;
                end
                iterator = [iterator SOM_neurons(x,y)];
            endfor;
        endfor;
        
        
        iterator
        for i = iterator;
            temp = SOM_weights(i,:) + ((alphaZero/(1+t/inputLength)) .* input(e,:));
            SOM_weights(i,:) = temp ./ norm(temp);            
        endfor;


        if (radius > 0)
            radius = radius - 1;
        endif;
    endfor;
endfor;



# SOM Supervised Learning
################################################################################
labels = zeros(neuronsX,  neuronsY);

for i = 1: neuronsX * neuronsY;

    dist = zeros(1,inputLength);
    for e = 1:inputLength;
        dist(e) = sum(abs(input(e,:) .* SOM_weights(i,:)));
    endfor;
    [M,I] = max(dist);
    [M,I] = max(expectedOutput(I,:));

    labels(i) = I;
endfor;
labels