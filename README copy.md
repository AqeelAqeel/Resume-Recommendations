<!-- ⚠️ This README has been generated from the file(s) "blueprint.md" ⚠️--><p align="center">
  <img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/logo-shadow.png" alt="Logo" width="150" height="150" />
</p>
<h1 align="center">@Resume Recommender</h1>
<p align="center">
		<a href="https://npmcharts.com/compare/@appnest/readme?minimal=true"><img alt="Downloads per month" src="https://img.shields.io/npm/dm/@appnest/readme.svg" height="20"/></a>
<a href="https://www.npmjs.com/package/@appnest/readme"><img alt="NPM Version" src="https://img.shields.io/npm/v/@appnest/readme.svg" height="20"/></a>
<a href="https://david-dm.org/andreasbm/readme"><img alt="Dependencies" src="https://img.shields.io/david/andreasbm/readme.svg" height="20"/></a>
<a href="https://github.com/andreasbm/readme/graphs/contributors"><img alt="Contributors" src="https://img.shields.io/github/contributors/andreasbm/readme.svg" height="20"/></a>
<a href="https://github.com/badges/shields"><img alt="Custom badge" src="https://img.shields.io/badge/custom-badge-f39f37.svg" height="20"/></a>
<a href="https://github.com/andreasbm/readme/graphs/commit-activity"><img alt="Maintained" src="https://img.shields.io/badge/Maintained%3F-yes-green.svg" height="20"/></a>
	</p>

<p align="center">
  <b>Automatically generate a list of keywords to improve your resume and the 5 active listings you are the greatest match for.</b></br>
  <sub>Use this readme generator to easily generate beautiful readme's like this one! Simply extend your <code>package.json</code> and create a readme blueprint. On Github, the README file is like the landing page of your website because it is the first thing visitors see. You want to make a good first impression.<sub>
</p>

<br />


<p align="center">
  <img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/demo.gif" alt="Demo" width="800" />
</p>

* **Simple**: Extremely simple to use - so simple that it almost feels like magic!
* **Powerful**: Customize almost everything - add your own templates and variables if you like
* **Awesome**: The tool you don't know you need before you have many different repositories that all need maintenance



[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)](#table-of-contents)

# ➤ Table of Contents

* [➤ Project Overview](#Project-Overview)
* [➤ Project Motivations](#Project-Motivations)
* [➤ The Dataset & EDA](#The-Dataset-&-EDA)
	* [Attaining Data](##Attaining-Data)
	* [The Dataset](##The-Dataset)
	* [Data Processing](#Data-Processing)
  * [Data Visuals](#Data-Visuals)
* [➤ The Model](#-templates)
	* [Performance Comparison](#Performance-Comparison)
	* [Outputs](#logo)
* [➤ Web App Dashboard](#-contributors)
	* [License](#license)
* [➤ Closing Remarks](#-license)
	* [Future Undertakings](#Future-Undertakings)
	* [Gratitude](#license)
  * [Contributors](#contributors)
	* [Reference Material](#Reading-Material)



[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)](#installation)

## ➤ Project Overview

```python
print("Hello Reader!")
```

Resume Reccomender leverages a Natural Language Processing technology stack in Python to match an input string of text (denoted as a "Resume") to needed keywords to better match job descriptions, as well as provide the URL of the top 5 closest job matches based on listing contents from the database. The database consists of over 6,000 job postings from Indeed.com that are currently active (as of August 2021) and was acquired through webscraper contained within this repo's `/src/jupyter-notebooks` folder. Project prepared by Aqeel Ali for the purpose and presentation of the final capstone project for Galvanize Data Science Full-time Immersive Bootcamp Remote Pacific Cohort #7. 

[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)](#getting-started-quick)

## ➤ Project Motivations

As a current full-time job seeker, Aqeel (author) was experiencing troubling low yield results on applications that appeared, at first glance, to be a great fit. 

Aqeel had discovered that candidates, no matter how qualified they are, slip through the cracks of all industries' mass adoption of Applicant Track Systems. Applicant Tracking Systems have gained massive popularity across multiple domains and use cases in organizational management, talent solutions, and much more. Aqeel realized him, and oter job seekers, were up against artificial intelligence programs designed to weed out seemingly less qualified or fit applicants. Job applications companies and hiring staff are inbounded with on a daily basis, and this tool serves a great purpose for expediting the hiring process in a time and cost efficient matter for companies.

Upon discovering that the ATS filtration systems base a lot of decision making power with the contents of resumes and cover letters, Aqeel set out to develop a tool for job seekers and the like to see how their resume could be improved with the addition of suggested keywords against many currently active job listings, and what job listings they are semantically the best fit for. 

[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)](#getting-started-slower)

# ➤ The Dataset & EDA

This getting started guide is a little bit longer, but will give you some superpowers. Spend a minute reading this getting started guide and you'll have the best README file in your town very soon.

## Attaining Data

The dataset consits of 9,055 rows of data across 6 labels. These labels, the job titles, serve as the classes that the model will be trained to predict. 

The 6 labels were:

```python
job_titles = 'Recruiter','Data Scientist', 'Financial Analyst', 'Physician', 'Underwriter', 'Chemical Engineer'
```

The scraper 
The scraper source code can be found in the `Indeed-Job_Scraper` file. 



## Job Dataset Value Counts

| Job Titles        | Quantity |
|-------------------|----------|
| Recruiter         | 2042     |
| Data Scientist    | 1597     |
| Financial Analyst | 1523     |
| Physician         | 1462     |
| Underwriter       | 1403     |
| Chemical Engineer | 1028     |



## Data Processing

Insert NLP Procedures here

NLP Procedures were used on the input text data to convert objects into vectors, dictionaries and symbols which can be handled very effectively using python library tools. Many operations such as searching, clustering, and keyword extraction were all done using very simple data structures, such as feature vectors.


Run the `node_modules/.bin/readme generate` command and a README file will be generated for you. If you want to go into depth with the readme command, check out the following options or write `node_modules/.bin/readme generate -h` in your terminal if that's your cup of tea.




## Data Visuals

Put charts and EDA stuff here

Plot distribution chart

PCA chart
[]
[]
[]

To configure this library you'll need to create a `blueprint.json` file. This file is the configuration for the templates we are going to take a look at in the next section. If you want to interpolate values from the configuration file into your README file you can simply reference them without a scope. Eg. if you have the field "link" in your `blueprint.json` you can write `{{ link }}` to reference it.

Great. Now that we have the basics covered, let's continue and see how you can use templates!

[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)](#templates)

# ➤ The Model

If you have come this far you are probably interested to figure out how to use README templates. This library comes with a set of pre-defined templates to make your readme awesome, but you can of course create your own. More about that later, let's not get ahead of our self just yet.

Utilized sklearn library built-in models to predict job titles given job descriptions. All of the models when hypertuned performed very strongly, with Logistic Regression being the top choice. 

## Performance Comparison

| RandomForestClassifier(max_depth=3, n_estimators=200)  | Precision | Recall | F1-Score |
|---------------------------------------------------------------------------------------------------------------|-----------|--------|----------|
| accuracy                                                                                                      |           |        | 0.86     |
| macro average                                                                                                 | 0.84      | 0.92   | 0.87     |
| weighted average                                                                                              | 0.89      | 0.86   | 0.86     |

| LinearSVC                                          | Precision | Recall | F1-Score |
|---------------------------------------------------------------------------------------------------------------|-----------|--------|----------|
| accuracy                                                                                                      |           |        | 0.95     |
| macro average                                                                                                 | 0.95      | 0.95   | 0.95     |
| weighted average                                                                                              | 0.95      | 0.95   | 0.95     |

| MultinomialNB                                        | Precision | Recall | F1-Score |
|---------------------------------------------------------------------------------------------------------------|-----------|--------|----------|
| accuracy                                                                                                      |           |        | 0.86     |
| macro average                                                                                                 | 0.84      | 0.92   | 0.87     |
| weighted average                                                                                              | 0.89      | 0.86   | 0.86     |

| Logistic Regression | Precision | Recall | F1-Score |
|---------------------------------------------------------------------------------------------------------------|-----------|--------|----------|
| accuracy                                                                                                      |           |        | 0.97     |
| macro average                                                                                                 | 0.96      | 0.96   | 0.97     |
| weighted average                                                                                              | 0.97      | 0.96   | 0.97     |

The models were cross validated across 10 K-Folds; 


## Outputs

Keyword suggestions

Job Listing recommendations


[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)](#table-of-contents)

# Web App Dashboard 

Deployed model using streamlit.io as a python-friendly front-end web development tool. 


## Results Display

Add content and screenshot of displaying results page

[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)](#table-of-contents)


# Closing Remarks



## Future Undertakings



## Gratitude 

## Reference Material

## Contributors

| [<img alt="Andreas Mehlsen" src="https://avatars1.githubusercontent.com/u/6267397?s=460&v=4" width="100">](https://twitter.com/andreasmehlsen) | [<img alt="You?" src="https://joeschmoe.io/api/v1/random" width="100">](https://github.com/andreasbm/readme/blob/master/CONTRIBUTING.md) |
|:--------------------------------------------------:|:--------------------------------------------------:|
| [Andreas Mehlsen](https://twitter.com/andreasmehlsen)

