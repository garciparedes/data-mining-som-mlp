
#
# Self-organizing map
#
# Sergio Garcia Prado
# garciparedes.me
#


# Config Values
################################################################################

inputFile = dlmread('digitos.entrena.normalizados.txt');
neuronsY = 6;
neuronsX = neuronsY * 2;



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
      RNA(i,j,k) = rand/2; 
    endfor;
    RNA(i,j,:) = RNA(i,j,:) ./ sqrt(sum(RNA(i,j,:).^2));
  endfor;
endfor;



# SOM Unsupervised Learning
################################################################################
#
for e = 1:1;
  result = zeros(neuronsX, neuronsY);
  for i = 1:neuronsX;
    for j = 1:neuronsY;
      suma = 0;
      for k = 1:inputDimens;
        suma = suma + (input(e,k) - RNA(i,j,k))^2;
      endfor;
      result(i,j) = sqrt(suma);
    endfor;
  endfor;
  result
  [M,I] = min(result(:));
  [I_row, I_col] = ind2sub(size(result),I);
  
  I_row
  I_col

  for x = I_row - floor(neuronsX /2)+1 : I_row + floor(neuronsX /2)-1;
    
    if (x < 1)
      x = x + neuronsX;
    elseif(x > neuronsX)
      x = x - neuronsX;
    end
    x
    
  endfor;
  
  for y = I_col - floor(neuronsY /2)+1 : I_col + floor(neuronsY /2)-1;
    if (y < 1)
      y = y + neuronsY;
    elseif(y > neuronsY)
      y = y - neuronsY;
    end
    y
  endfor;
    
endfor;
