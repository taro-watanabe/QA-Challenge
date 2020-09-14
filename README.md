# QA Challenge  



## Environment

- Operating System - Ubuntu 20
- Dataset - Provided file from link
- Coding Environment - Pycharm Community - Python 3.7/3.8 Venv



## Brief Workflow

0. Research on Docker + Flask API  *Day 1-2*

1. Docker : Installation of docker + docker-compose *Day3*
2. Docker : Pulling from `localhost:5000` *Day4*

--------

3. Python Environment Setup : Request to `localhost:5000` *Day4*
4. Python Dataframe : JSON -> Pandas Dataframe Conversion *Day 5*
5. Python Analysis : Machine Learning models, etc. *Day5-7*
6. Python UI : Migrating & Local Hosting WebApp (Streamlit) *Day 7*



## Information

I failed to integrate the Python file into the docker system, as port opening on `8501` failed due to (probably) a simple error. Thus the docker environment has not been modified at all.  Simply `cd` to the additional python file `stream.py` and `streamlit run stream.py` should launch the local port at `8501`.
Also due to the lack of knowledge on docker/SQL, I was unable to complete the bonus section. Since `streamlit` is ran through a local port, the storage must be doable.



The `Streamlit` is constructed in the following way:

- Dataset size decision for optional faster calculation time
- Choice between the two main tasks
- For the first task, for a visualization capability testing purpose, a live, interactive hyperparameter editable model comparison is available, although it still has some bugs to be remained.
- On both tasks, a pipeline of 5 models has been created, and for wither cases, some intuition is provided with either accuracy score or the regression plane visualization.
- The thought process of creating the pipeline is written further in the `ipynb` file,  please consult that file for more writings. They also have codes for intuition that are simply not present in the final `py` file. Note that this `ipynb` file stops at task 1, as task 2 is most a conversion into a regression model, making it redundant.

All of the packages were install through pip as of 14/09/2020.



## General Comments 



The Machine Learning Part gave me some more space to breathe, as I have done something in the past, however, due to lack of experience in virtual environments, API requests (Flask), and Web UI creation, I am unsatisfied with how much I was unable to perfect the system. If I were to improve (given more experience and time), I would: 



- Visualization of classification decision boundaries
- Docker Integration
- Consistent Features on both tasks - Interactive UI for task 2 - Visualization for task 1
- General reduction error
- Time optimization
- Request Optimization



That being said, it was very enjoyable to work on this test, and regardless of the selection result, I will thank you for the opportunity for me to learn.



