# Discuit

Dynamic item set clustering UI tool

Goal: Splitting datasets (e.g. words defined by several variables) into subsets that are as comparable as possible.


The commandline tool at the moment takes a csv file as input and generates a defined number of matched sets for a given number of continuous and categorical variables. One of the categorical variables can be selected to be split absolutely even across sets.

This will be integrated in an GUI. So far there is a basic model (model.py) and some first attempts at a GUI with PyQt.

## To do model
- remove hard-coding of input etc and provide them as arguments OR tackle this in the GUI directly
- add documentation to code

## To do GUI
- finish interface design
- integrate splitting code 


Image by [Clker-Free-Vector-Images](https://pixabay.com/users/clker-free-vector-images-3736/?utm_source=link-attribution&amp;utm_medium=referral&amp;utm_campaign=image&amp;utm_content=304801) from [Pixabay](https://pixabay.com/?utm_source=link-attribution&amp;utm_medium=referral&amp;utm_campaign=image&amp;utm_content=304801)

## Running the script
In the terminal go to the folder with the script run the following: 
python3 model.py "name of input file" [number of desired sets] --columns l/c/n/d

Example: python3 model.py test-files/dr-final.csv 2 --columns l c n n n c c c c c c c c a c c c c c c d

The input file needs to be a .csv file with a first line containing headings followed by rows that 
represent the different items. Each column specifies one variable. There cannot be missing data (for now).
When launching the script, please specify per column what kind a data the script should expect:
(l)abel: just a label, will not be taken into consideration, could be the itemname or itemnumber. This can only be assigned once.
(n)umerical: a numerical variable, such as frequency or AoA
(c)ategorical: a categorical variable, such as "transitivity" or "accuracy"
(a)bsolute: this needs to be perfectly divided between sets. This can only be assigned once.
(d)isregard: a column that does not need to be taken into account for the split, but contains other information you have in the same file.

The script will try 20 times to come-up with a good split. If it doesn't it will give up and output it's last try.
You can always run it again. Often it will succeed eventually.

If you run the script without specifying --columns, you will be asked what you want per column.




