
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
neuronsY = 8;
neuronsX = 12;
times = 50;
alphaZero = 25;

# Input Normalization
################################################################################

input = inputFile;
input([2:2:size(input,1)],:) = [];
input = [input ones(size(input, 1), 1)];
input = input ./ sqrt(sum(input.^2,2));


inputLength = size(input,1);
inputDimens = size(input,2);

#inputLength = 10;


expectedOutput = inputFile;
expectedOutput(:,[11:size(expectedOutput,2)]) = [];
expectedOutput([1:2:size(expectedOutput,1)],:) = [];



# SOM Inicialization
################################################################################

RNA = zeros(neuronsX, neuronsY, inputDimens);
for i = 1:neuronsX;
    for j = 1:neuronsY;
        for k = 1:inputDimens;
            RNA(i,j,k) = rand -0.5;
        endfor;
        norm = sqrt(sum(RNA(i,j,:).^2));
        RNA(i,j,:) = RNA(i,j,:) ./ norm;
    endfor;
endfor;



# SOM Unsupervised Learning
################################################################################
radius = min(floor(neuronsX /2), floor(neuronsY /2));

for t = 1:times;
    for e = 1:inputLength;

        # Get Distance from input to neuron
        distances = zeros(neuronsX, neuronsY);
        for i = 1:neuronsX;
            for j = 1:neuronsY;

                
                distances(i,j) = 0;
                for k = 1:inputDimens;
                    distances(i,j) = distances(i,j) + (input(e,k) * RNA(i,j,k));
                endfor;
                
            endfor;
        endfor;
        [M,I] = max(distances(:));
        [xWin, yWin] = ind2sub(size(distances),I);
        #[e, xWin, yWin]
        #distances
        
        
        # Update weights of neurons
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
                
                temp = zeros(inputDimens);
                for k = 1 : inputDimens;
                    temp(k) = (RNA(x,y,k) + (alphaZero/(1+t/inputLength)) .* input(e,k));
                endfor;
                
                norm = sqrt(sum(temp(:).^2));
                for k = 1 : inputDimens;
                    RNA(x,y,k) = temp(k) ./ norm;
                endfor;

            endfor;
        endfor;
        if (radius > 0)
            radius = radius - 1;
        endif;
    endfor;
endfor;



# SOM Supervised Learning
################################################################################
labels = zeros(neuronsX, neuronsY);

for i = 1:neuronsX;
    for j = 1:neuronsY;
        
        dist = zeros(1,inputLength);
        for e = 1:inputLength;
       
            for k = 1:inputDimens;
                dist(e) = dist(e) + abs(input(e,k) * RNA(i,j,k));
            endfor;
            
            [M,I] = min(dist);
            [M,I] = max(expectedOutput(I,:));
            labels(i,j) = I;
        endfor;
    endfor;
endfor; 
labels
