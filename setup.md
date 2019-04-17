---
title: Setup
---
# Software Packages Required

You will need to have an installation of Python 3 with the matplotlib, pandas, numpy and optionally opencv packages. 

The [Anaconda Distribution](https://www.anaconda.com/distribution/#download-section) includes all of these except opencv by default.

## Installing OpenCV with Anaconda

* Load the Anaconda Navigator
* Click on "Environments" on the left hand side.
* Choose "Not Installed" from the pull down menu next to the channels button.
* Type "opencv" into the search box.
* Tick the box next to the opencv package and then click apply. 

## Installing from the Anaconda command line

From the Anaconda terminal run the command `conda install -c conda-forge opencv`

# Download the data

Please create a sub directory called data in the directory where you save any code you write.

Download the the following files to this directory:

* [Gapminder Life Expectancy Data](data/gapminder-life-expectancy.csv)
* [World Bank GDP Data](data/worldbank-gdp.csv)
* [World Bank GDP Data with outliers](data/worldbank-gdp-outliers.csv)

If you are using a Mac or Linux system the following commands will download this:

~~~
mkdir data
cd data
wget https://scw-aberystwyth.github.io/machine-learning-novice/data/worldbank-gdp.csv
wget https://scw-aberystwyth.github.io/machine-learning-novice/data/worldbank-gdp-outliers.csv
wget https://scw-aberystwyth.github.io/machine-learning-novice/data/gapminder-life-expectancy.csv
~~~
{: .bash}

{% include links.md %}
