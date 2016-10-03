neuronsX = 10;
neuronsY = neuronsX * 2;

inputFile = dlmread('digitos.entrena.normalizados.txt');

input = inputFile;
input([2:2:size(input,1)],:) = [];
input = [input ones(size(input, 1), 1)];


expectedOutput = inputFile;
expectedOutput(:,[11:size(expectedOutput,2)]) = [];
expectedOutput([1:2:size(expectedOutput,1)],:) = [];



normalizedInput = input./ sqrt(sum(input.^2,2));

RNA = zeros(neuronsX, neuronsY, size(input,1));
for i = 1:10;
  for j = 1:20;
    for k = 1:size(input,1);
      RNA(i,j,k) = rand; 
    endfor;
  endfor;
endfor;
