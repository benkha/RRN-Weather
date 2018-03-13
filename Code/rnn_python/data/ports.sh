#!/bin/bash
# mv CS294Data/*GCK*.pkl GCK/
for i in $( cat ports.txt ); do
    mkdir $i
    mv "CS294Data/*$i*.pkl" $i/
done
