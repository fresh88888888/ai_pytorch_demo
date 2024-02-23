#!/bin/sh

echo "You can list numbers and text like this: "

for n in 1 2 3 four
do
    echo "Number $n"
done

echo "Or specify a range of numbers:"

for n in {1..5}
do
    echo "Number $n"
done

echo "Or use the output of another command."
for f in $(ls)
do
    echo $f
done

#--------------------------------------#

for i in {1..10}
do
    if test $i -eq 3
    then
        echo "I found the 3."
    else
        echo "Not looking for the $i."
    fi
done

x=99;
if [ $x > 50 ] ; then
    echo "too high";
elif [ $x < 50 ] ; then
    echo "too low";
else
    echo "spot on";
fi

