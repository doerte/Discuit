# Discuit

Dynamic item set clustering UI tool

Goal: Splitting datasets (e.g. words defined by several variables) into subsets that are as comparable as possible.


The commandline tool at the moment takes a csv file as input and generates a defined number of matched sets for a given number of continuous and 1 categorical variable

This will be integrated in an GUI. So far there is a basic model (model.py) and some first attempts at a GUI with PyQt.

## To do model
- update splitting code to include more than 1 categorical variable
- add option to have an "absolute split" for one variable 
- remove hard-coding of input etc and provide them as arguments OR tackle this in the GUI directly
- add documentation to code

## To do GUI
- finish interface design
- integrate splitting code 


Image by [Clker-Free-Vector-Images](https://pixabay.com/users/clker-free-vector-images-3736/?utm_source=link-attribution&amp;utm_medium=referral&amp;utm_campaign=image&amp;utm_content=304801) from [Pixabay](https://pixabay.com/?utm_source=link-attribution&amp;utm_medium=referral&amp;utm_campaign=image&amp;utm_content=304801)
