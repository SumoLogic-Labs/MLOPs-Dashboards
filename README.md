# ML Observability

Sumologic dashboards for observing your ML data and experimentation logs that enables you to monitor your model performance in real time. We provide preconfigured dashboards that have been bundled under different family of machine learning models which include information about model state and data distribution, along with industry-wide used metrics for performance evaluation, all generated at runtime.



1. **_Set once, Monitor Anytime Anywhere: Your Data, Your Control_**
2. No need to change your way of logging, **_USE FERs INSTEAD_**


# Prerequisites



1. The data is recommended to follow a json formatting. If not, you need to use FERs to parse out the predefined features

    NOTE: We do not want to restrict how you should format your logs. While it is easier to dump everything as a json, you would still be fine if you define a Feature Extraction Rule to get the feature.

2. _sourceCategory should be set to ml_logs


## Required Features

To be able to analyze these metrics, the first step is to get the metrics into Sumo. We have organized the model dashboards under the following categories according to well-known ML model/problem families: **_Classification Models, Regression Models, LLM Models_**. And for each of the above, we have further segmented the exploration under



1. Model Experiment Tracking
2. Model runtime metrics Tracking

For each of the above model types, we define a schema in which the data is recommended to be ingested. The required features for Model experimentation is as follows:


<table>
  <tr>
   <td>
   </td>
   <td>Classification
   </td>
   <td>Regression
   </td>
   <td>LLM
   </td>
  </tr>
  <tr>
   <td>Model Experiment
   </td>
   <td><strong><em>Training metadata exp_id,</em></strong>
<p>
<em>model_name, \
hyperparameters, duration, </em>
<p>
<em>in_memory, </em>
<p>
<em>model_size, </em>
<p>
<em>CPU</em>
<p>
<em>—-----------------------------<strong>Validation Data </strong></em>
<p>
<em>exp_id, </em>
<p>
<em>expected, </em>
<p>
<em>predicted,</em>
   </td>
   <td><strong><em>Training metadata exp_id,</em></strong>
<p>
<em>model_name, \
hyperparameters, duration, </em>
<p>
<em>in_memory, </em>
<p>
<em>model_size, </em>
<p>
<em>CPU</em>
<p>
<em>—------------------------------</em>
<p>
<strong><em>Validation Data</em></strong>
<p>
<em>exp_id, </em>
<p>
<em>expected, </em>
<p>
<em>predicted,</em>
   </td>
   <td><strong><em>Training Metadata</em></strong>
<p>
<em>exp_id, \
parameters, </em>
<p>
<em>duration, </em>
<p>
<em>epoch, </em>
<p>
<em>accuracy, </em>
<p>
<em>loss</em>
<p>
<em>—--------------------------<strong>Validation Data</strong></em>
<p>
<em>exp_id, </em>
<p>
<em>input, </em>
<p>
<em>output, </em>
<p>
<em>bleu_score, rouge_score, </em>
<p>
<em>toxicity, </em>
<p>
<em>perplexity</em>
   </td>
  </tr>
  <tr>
   <td>Real Time Analysis
   </td>
   <td><em>expected, </em>
<p>
<em>predicted,</em>
   </td>
   <td><em>expected, </em>
<p>
<em>predicted,</em>
   </td>
   <td>NA
   </td>
  </tr>
</table>


Note that most of these features are curated based on industry practice but are not limited to what you can have. 

### Setup for data ingestion:

1.  Format your logs to follow the schema described above. Eg.

        {   
        "exp_id": "TH67RDF",
        "model_name": "KNN",
        "hyperparameters": "{'algorithm': 'auto', 'leaf_size': 30, 'k':12}",
        "duration": 126,
        "in_memory": 40,
        "model_size": 1.25,
        "CPU": 11.2
        }


    If your logs follow a different formatting, FERs can be used to extract the defined features.


2. Setup a collector (Installed/Host) and a source to get your logs in Sumo


3. Download the json template that matches your use case from the open source Sumo projects 


And you are all set to get started. We have added details about all the dashboards we offer below. 

Broadly, there are three families of  metrics in these phases:



1. Model Experiment Tracking:(Train-Test-Predict)
    1. Model tuning metrics: during Model: Train-Test, we emphasize hyperparameter tuning that impact  training duration, model size, CPU Utilization, performance evaluation metric
    2. Model validation metrics: based on the model family (classification, regression or LLM), we suggest evaluating models based on true/false positive rate, forecast accuracy or other appropriate measurement.
2. Model runtime metrics Tracking(Deploy & Observe) : once deployed, the production model is evaluated for model drift and its underlying root causes such as distribution shifts in production.

We expand on these metrics for classification, regression and LLM models below and point out the nuances that led particular recommendations for metrics.

Classification Models:

Classification models  predict the categorical labels of new instances based on past observations. These models learn from labeled training data, where each data point has a known category or class. The goal is to build a model that can generalize well to classify unseen data accurately. 

This Dashboard provides deep insights into 


1. **Model tuning Metrics**

A wide table captures metrics during training with easy to order experimental runs based metrics of choice. The recommended metrics to track are model hyperparameters, training duration, model size, CPU Utilized, RAM used etc. This section of the dashboard helps decide which model set would work the best based on the operational model metrics. 

2. **Model Validation Metrics**

These are metrics generated using the expected and predicted values while highlighting best performing metrics in each category(values highlighted in green, blue and yellow blocks are best performing metrics). Metrics generated are 

1. Model Accuracy
2. True positive Rate
3. False positive Rate

These metrics are built into the dashboards. These metrics are compared against all the experiments that were run together to give brief insight into how the model is going to perform on unseen data. These metrics are useful for making model performance related decisions

3. **Model Runtime evaluation metrics**


At runtime, it gives you the class distribution of both expected and predicted value for better understanding into where the model predictions are leaning more towards. Assuming a threshold, the evaluation metrics can be used directly to set up monitors and get alerts for model retraining.


### Regression Models:

Regression models are a type of machine learning model used for predicting continuous values rather than categorical class labels. These models are trained on data where the target variable is a real number, and the goal is to establish a relationship between the input features and the output.

This Dashboard provides deep insights into 



1. **Model tuning Metrics:**

A wide table captures metrics during training with easy to order experimental runs based metrics of choice. The recommended metrics to track are model hyperparameters, training duration, model size, CPU Utilized, RAM used etc

2. **Model Validation Metrics**


These are metrics generated using the expected and predicted values while highlighting best performing metrics in each category(values highlighted in green, purple and yellow blocks are best performing metrics). Metrics generated are 

1. Mean Absolute error
2. Mean Squared error
3. Root Mean Squared error
4. Residuals

These metrics are built into the dashboards. These metrics are compared against all the experiments that were run together to give brief insight into how the model is going to perform on unseen data.

3. **Model Runtime evaluation metrics**

At runtime evaluation, we do a side by side comparison of the distribution of observed data and model predictions over metrics such as min, max, average and standard deviation while generating model evaluation metrics(as described in point 2) to understand model performance over time. The capturing of metrics - residuals over time helps in identifying **drifts** in model performance and data distribution which can be added to alerting monitors to keep getting notified without having the need to check the dashboards again and again.



### LLM Models:

Large Language Models (LLMs) refer to advanced natural language processing models that have a massive number of parameters, enabling them to process and generate human-like text across a wide range of tasks. These models are typically based on deep learning architectures, specifically transformer architectures. 

As such the Model:Train-Test stage really refers to selecting a language model from a roster of open source and commercial offerings and evaluating their efficacy and tuning them through experiments. This dashboard can encompass classification cases as well, the highlight being, we can deep dive into metrics that are useful for making decisions about the Large Language model itself.



This dashboard captures:



1. A wide table capturing metrics from during training with easy to order experimental runs and choose which language model works the best based on metric
    1. Accuracy
    2. Validation loss
    3. Duration

    These metrics are calculated for the complete training and individual epochs with best performing metric highlighted on the top for easy decision making

2. Model validation metadata captured along with metrics such as BLEU score and ROUGE score with an option to add additional metrics where needed. 

    BLEU score and ROUGE score are numbers between zero and one that measures the similarity of the machine-translated text to a set of high quality reference translations