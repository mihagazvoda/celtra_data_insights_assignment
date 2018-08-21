~~~markdown
# Celtra Data Insights assignment

My solution for the Celtra Data Insights assignment. The solution is implemented in Python. Files to run:  
* celtra_task-notebook: Jupyter Notebook containing the tasks 1-4.2 and the discription for all tasks (including 4.3)
* er_live_tracking: Python file for ER live tracking dashboard implemented in Dash 

## Installation
Copy this repository to your computer
```
# get this repository
git clone https://github.com/mihagazvoda/celtra_data_insights_assignment.git
cd celtra_data_insights_assignment
```

Having [Anaconda](https://www.continuum.io/downloads) or  installed simply create your ENV with 

```
# install working environment with conda
conda env create -n celtra_task -f environment.yml

# environment should be activated now
# if not type: source activate celtra_task

# start juypter notebook if you want to run the celtra_task-notebook
jupyter notebook

# run ER live tracking dashboard
python er_live_tracking.py
# and open localhost: 
# http://127.0.0.1:8050/
```

  
Alternatively you can also create a python virtual enviroment and 
```
pip install -r requirements.txt
```







~~~