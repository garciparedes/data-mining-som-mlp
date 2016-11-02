#!/bin/bash

echo '--------------------------------------------------------------------------------'
echo 'Title: SOM + MLP implementation comparative'
echo 'Subject: Data Mining'
echo 'Author: Sergio García Prado (garciparedes.me)'
echo '--------------------------------------------------------------------------------'
echo 'Octave:'
echo
octave som.m
echo '--------------------------------------------------------------------------------'
echo
echo
echo '--------------------------------------------------------------------------------'
echo 'Weka:'
echo
javac -classpath "lib/*" WekaMultiLayerPerceptron.java
java -classpath "lib/*:." WekaMultiLayerPerceptron
echo '--------------------------------------------------------------------------------'
