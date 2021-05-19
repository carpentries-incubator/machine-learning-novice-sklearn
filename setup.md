---
title: "Setup"
---

## Download the data

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

## JupyterLab and scientific Python packages

In order to follow the presented material, you should launch the JupyterLab
server in the root directory. To see an explanation on how to use JupyterLab, refer to the
[first episode of the Software Carpentry Python Gapminder lesson][gapminder-jupyter]. Essentially, you can navigate
to the lesson directory and start JupyterLab with the following command:
```
jupyter-lab
```
{: .language-bash}


JupyterLab and the other scientific packages we are going to use (NumPy, Pandas, Matplotlib) 
usually come by default with Anaconda. Optionally, we can also install the OpenCV package with the following command:
```
conda install opencv -c conda-forge
```
{: .language-bash}

or you can install it with Anaconda:

* Load the Anaconda Navigator
* Click on "Environments" on the left hand side.
* Choose "Not Installed" from the pull down menu next to the channels button.
* Type "opencv" into the search box.
* Tick the box next to the opencv package and then click apply. 

**If you don't have Anaconda installed, follow the video instructions below.**

## Installing Python Using Anaconda

{% include python_install.html %}

<br>

[anaconda]: https://www.anaconda.com/
[anaconda-mac]: https://www.anaconda.com/download/#macos
[anaconda-linux]: https://www.anaconda.com/download/#linux
[anaconda-windows]: https://www.anaconda.com/download/#windows
[gapminder]: https://en.wikipedia.org/wiki/Gapminder_Foundation
[jupyter]: http://jupyter.org/
[python]: https://python.org
[video-mac]: https://www.youtube.com/watch?v=TcSAln46u9U
[video-windows]: https://www.youtube.com/watch?v=xxQ0mzZ8UvA
[gapminder-jupyter]: https://swcarpentry.github.io/python-novice-gapminder/01-run-quit/index.html
