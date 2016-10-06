
#
# Self-organizing map
#
# Sergio Garcia Prado
# garciparedes.me
#


# Config Values
################################################################################

inputFile = dlmread('digitos.entrena.normalizados.txt');
neuronsY = 8;
neuronsX = 12;
times = 10;


# Input Normalization
################################################################################

input = inputFile;
input([2:2:size(input,1)],:) = [];
input = [input ones(size(input, 1), 1)];
input = input ./ sqrt(sum(input.^2,2));


inputLength = size(input,1);
inputDimens = size(input,2);


#{
expectedOutput = inputFile;
expectedOutput(:,[11:size(expectedOutput,2)]) = [];
expectedOutput([1:2:size(expectedOutput,1)],:) = [];
#}



# SOM Inicialization
################################################################################

RNA = zeros(neuronsX, neuronsY, inputDimens);
for i = 1:neuronsX;
    for j = 1:neuronsY;
        for k = 1:inputDimens;
            RNA(i,j,k) = rand -0.5;
        endfor;
        RNA(i,j,:) = RNA(i,j,:) ./ sqrt(sum(RNA(i,j,:).^2));
    endfor;
endfor;



# SOM Unsupervised Learning
################################################################################
radius = min(floor(neuronsX /2), floor(neuronsY /2));

for t = 1:times;
    for e = 1:inputLength;

        # Get Distance from input to neuron
        result = zeros(neuronsX, neuronsY);
        for i = 1:neuronsX;
            for j = 1:neuronsY;

                result(i,j) = 0;
                for k = 1:inputDimens;
                    result(i,j) = result(i,j) + input(e,k)*RNA(i,j,k);
                endfor;

            endfor;
        endfor;
        [M,I] = max(result(:));
        [xWin, yWin] = ind2sub(size(result),I);

        # Update weights of neurons
        for x = (xWin - radius+1) : (xWin + radius-1);
            if (x < 1)
                x = x + neuronsX;
            elseif(x > neuronsX)
                x = x - neuronsX;
            end

            for y = (yWin - radius+1) : (yWin + radius-1);
                if (y < 1)
                    y = y + neuronsY;
                elseif(y > neuronsY)
                    y = y - neuronsY;
                end

                RNA(x,y,:) = (RNA(x,y,:) + (25/(1+t/inputLength)) * input(e,k)) ./ sqrt(sum((RNA(x,y,:) + (25/(1+t/inputLength)) * input(e,k)).^2));

            endfor;
        endfor;
        if (radius > 1)
            radius = radius - 1;
        endif;
    endfor;
endfor;

result
