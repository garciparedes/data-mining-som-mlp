
#
# Self-organizing map
#
# Sergio Garcia Prado
# garciparedes.me
#
clear all

function selfOrganizingMap(filename ='digitos.entrena.normalizados.txt',
     neuronsX = 8, neuronsY = 12, seasons = 50, alphaZero = 25)

    [input, expectedOutput, inputDimens, inputLength] = importFromFile(filename);
    

    # Input Normalization
    ################################################################################
    ################################################################################





    # SOM Inicialization
    ################################################################################
    ################################################################################

    SOM_weights = somInit(neuronsX, neuronsY, inputDimens);



    # SOM Unsupervised Learning
    ################################################################################
    ################################################################################

    radius = min(floor(neuronsX /2), floor(neuronsY /2));
    for t = 1:seasons;
        for e = 1:inputLength;
            distances = input(e,:) * SOM_weights';
            
            [M,I] = max(distances); 
            [xWin,yWin] = ind2sub([neuronsX, neuronsY],I);
            
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
                    iterator = [iterator sub2ind([neuronsX, neuronsY], x,y)];
                endfor;
            endfor;
            
            
            for i = iterator;
                temp = SOM_weights(i,:) + ((alphaZero/(1+t/inputLength)) .* input(e,:));
                SOM_weights(i,:) = temp ./ norm(temp);            
            endfor;


            if (radius > 0)
                radius = radius - 1;
            endif;
        endfor;
    endfor;


        
    savedOutput = zeros(inputLength, size(SOM_weights,1) +1);
    for e = 1:inputLength;

        # Get Distance from input to neuron
        distances = input(e,:) * SOM_weights';

        
        [M,I] = max(expectedOutput(e,:));

        
        savedOutput(e,:) =  [distances.^4 I];

    endfor;
    csvwrite('digitos.entrena.normalizados.output.csv', [(1:size(SOM_weights,1)), 9999; savedOutput]);

    #{
    # SOM Supervised Learning
    ################################################################################
    ################################################################################

    labels = zeros(neuronsX,  neuronsY);
    for i = 1: neuronsX * neuronsY;

        dist = zeros(1,inputLength);
        for e = 1:inputLength;
            dist(e) = sum((input(e,:) .* SOM_weights(i,:)));
        endfor;
        
        [M,I] = max(dist);
        [M,I] = max(expectedOutput(I,:));

        labels(i) = I;
    endfor;
    labels


    # SOM Test
    ################################################################################
    ################################################################################


    # SOM Test - Training Set
    ################################################################################

    success = 0;
    for e = 1:inputLength;

        # Get Distance from input to neuron
        distances = input(e,:) * SOM_weights';

        
        [M,I] = max((distances)); 
        [xWin,yWin] = ind2sub([neuronsX, neuronsY],I);
        
        [M,I] = max(expectedOutput(e,:));

        if( I == labels(xWin, yWin));
            success = success +1;
        endif;

    endfor;
    successRate = (success/inputLength)




    # SOM Test - Test Set
    ################################################################################

    inputFile = dlmread('digitos.test.normalizados.txt');

    input = inputFile;
    input([2:2:size(input,1)],:) = [];
    input = [input ones(size(input, 1), 1)];
    input = input ./ sqrt(sum(input.^2,2));


    inputLength = size(input,1);
    inputDimens = size(input,2);


    expectedOutput = inputFile;
    expectedOutput(:,[11:size(expectedOutput,2)]) = [];
    expectedOutput([1:2:size(expectedOutput,1)],:) = [];

    success = 0;
    for e = 1:inputLength;

        distances = input(e,:) * SOM_weights';  
        
        [M,I] = max((distances)); 
        [xWin,yWin] = ind2sub([neuronsX, neuronsY],I);
        
        [M,I] = max(expectedOutput(e,:));

        if( I == labels(xWin, yWin));
            success = success +1;
        endif;
    endfor;
    successRate = (success/inputLength)
    #}
endfunction;



function SOM_weights = somInit(neuronsX, neuronsY, inputDimens)
    SOM_weights = rand(neuronsX * neuronsY, inputDimens) - 0.5;
    for i = 1 : (neuronsX * neuronsY);
        SOM_weights(i,:) = SOM_weights(i,:) ./ norm(SOM_weights(i,:));
    endfor;
endfunction;

function [input, output, fdimens, fsize ] = importFromFile(filename)
    
    inputFile = dlmread(filename);

    input = inputFile;
    input([2:2:size(input,1)],:) = [];
    input = [input ones(size(input, 1), 1)];
    input = input ./ sqrt(sum(input.^2,2));

    fdimens = size(input,2);
    fsize = size(input,1);


    output = inputFile;
    output(:,[11:size(output,2)]) = [];
    output([1:2:size(output,1)],:) = [];
endfunction;




selfOrganizingMap()