from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
from plotly.graph_objs import Scatter
import os
import json
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import pickle

def home(request):
    world = pd.read_csv('carbonemission/meatcattleworldco.csv')
    data = dict(
        type = 'choropleth',
        colorscale='Reds',
        # reversescale = True,
        locations = world['Area'],
        locationmode = "country names",
        z = world['Value'],
        text = world['Area'],
        colorbar = {'title' : 'CO2 Emission in Meat Industry'},
        # wscale=False
        # autocolorscale=True
      ) 

    layout = dict(title = 'CO2 Emission in Meat Industry',margin={"r":35,"t":35,"l":35,"b":35},
                geo = dict(showframe = False,projection = {'type':'natural earth'})
             )
    choromap = go.Figure(data = [data],layout = layout)
    
    
    #fig5 = iplot(choromap)
    fig5 = choromap.to_html()
    df2000 = pd.read_csv('carbonemission/dataset2000.csv')
    df2017 = pd.read_csv('carbonemission/dataset2017.csv')
    fig6 = make_subplots(1, 2, specs=[[{'type':'domain'}, {'type':'domain'}]],
                        subplot_titles=['2000', '2017'])
    fig6.add_trace(go.Pie(labels=df2000['Item'], values=df2000['Value'], scalegroup='one',
                        name="CO2 2000"), 1, 1)
    fig6.add_trace(go.Pie(labels=df2017['Item'], values=df2017['Value'], scalegroup='one',
                        name="CO2 2000"), 1, 2)
    fig6.update_layout(title_text='Carbon Emission in Meat Industry')
    plot_div6 = plot(fig6, output_type='div', include_plotlyjs=False)
    context = {'fig5': fig5,'plot_div6': plot_div6}    
    return render(request,'index.html', context)

def input_visualize(request):    
    return render(request,'input_visualize.html')
def A_InputVisualize(request):    
    return render(request,'A_InputVisualize.html')
def N_InputVisualize(request):    
    return render(request,'N_InputVisualize.html')
def input_predict(request):    
    return render(request,'input_predict.html')
def solution(request):    
    return render(request,'solution.html')
def input_Compare(request):    
    return render(request,'input_Compare.html')



def pre(x,model_prophet,model_arima,model_neural,pitem,years):
    dffao = pd.read_csv(x)
    dffao.drop(['Domain','Area','Element','Item'],axis=1,inplace=True)
    dffao['Year'] = dffao['Year'].astype(str) + '/12/31'
    dffao['Year'] = dffao['Year'].str.replace('/','-')
    dffao['Year'] = pd.to_datetime(dffao['Year'])
    #arima
    df1 = pd.read_csv(model_arima)
    df1 = pd.DataFrame(df1.loc[df1['Item'] == pitem])
    df =  pd.DataFrame(df1[:57+years])
    df2 =  pd.DataFrame(df1[:57])
    df2.rename(columns = {'Emission Intensity':'actual'}, inplace = True)
    mape_a,rmse_a = forecast_accuracy(df2.actual,dffao.Value)
    #prophet
    df3 = pd.read_csv(model_prophet)
    df3 = pd.DataFrame(df3.loc[df3['Item'] == pitem])
    df5 =  pd.DataFrame(df3[:57+years])
    df4 =  pd.DataFrame(df3[:57])
    df4.rename(columns = {'Emission Intensity':'actual'}, inplace = True)
    mape_p,rmse_p = forecast_accuracy(df4.actual,dffao.Value)
    #neural
    df6 = pd.read_csv(model_neural,parse_dates=True)
    df6 = pd.DataFrame(df6.loc[df6['Item'] == pitem])
    df7 =  pd.DataFrame(df6[:57+years])
    df8 =  pd.DataFrame(df6[:57])
    df8.rename(columns = {'yhat1':'actual'}, inplace = True)
    mape_np,rmse_np = forecast_accuracy(df8.actual,dffao.Value)
    acc_list=[mape_p,rmse_p,mape_a,rmse_a,mape_np,rmse_np]
    df['Year'] = pd.to_datetime(df['Year'])
    df5['Year'] = pd.to_datetime(df5['Year'])
    #df7['ds'] = pd.to_datetime(df7['ds'])
    return df,df5,df7,acc_list

def preprocessing(x):
    dff = pd.read_csv(x)
    dff.drop(['Domain','Area','Element','Item'],axis=1,inplace=True)
    dff['Year'] = dff['Year'].astype(str) + '/12/31'
    dff['Year'] = dff['Year'].str.replace('/','-')
    dff['Year'] = pd.to_datetime(dff['Year'])
    return dff

def forecast_accuracy(forecast, act):
    mape = np.mean(np.abs(forecast - act)/np.abs(act))  # MApe
    rmse = np.mean((forecast - act)**2)**.5  # RMSE
    return mape,rmse

def getPredictions(x,years):
    if x == 'cereals':
        final,prophet,model_neural,acc_list=pre('carbonemission/indiaDataset/cereals_excluding_rice_old.csv','carbonemission/combined_data.csv',"carbonemission/combined_data_arima.csv","carbonemission/combined_data_neural_prophet.csv","Cereals Exc Rice",years)
        final1=pd.DataFrame(preprocessing('carbonemission/indiaDataset/cereals_excluding_rice_old.csv'))
    elif x== 'MeatBuffalo':
        final,prophet,model_neural,acc_list=pre('carbonemission/indiaDataset/meat_buffalo_old.csv','carbonemission/combined_data.csv',"carbonemission/combined_data_arima.csv","carbonemission/combined_data_neural_prophet.csv","Meat Buffalo",years)
        final1=pd.DataFrame(preprocessing('carbonemission/indiaDataset/meat_buffalo_old.csv'))
    elif x== 'MeatCattle':
        final,prophet,model_neural,acc_list=pre('carbonemission/indiaDataset/meat_cattle_old.csv','carbonemission/combined_data.csv',"carbonemission/combined_data_arima.csv","carbonemission/combined_data_neural_prophet.csv","Meat Cattle",years)
        final1=pd.DataFrame(preprocessing('carbonemission/indiaDataset/meat_cattle_old.csv'))
    elif x== 'MeatChicken':
       final,prophet,model_neural,acc_list=pre('carbonemission/indiaDataset/meat_chicken_old.csv','carbonemission/combined_data.csv',"carbonemission/combined_data_arima.csv","carbonemission/combined_data_neural_prophet.csv","Meat Chicken",years)
       final1=pd.DataFrame( preprocessing('carbonemission/indiaDataset/meat_chicken_old.csv'))
    elif x== 'MeatGoat':
        final,prophet,model_neural,acc_list=pre('carbonemission/indiaDataset/meat_goat_old.csv','carbonemission/combined_data.csv',"carbonemission/combined_data_arima.csv","carbonemission/combined_data_neural_prophet.csv","Meat Goat",years)
        final1=pd.DataFrame(preprocessing('carbonemission/indiaDataset/meat_goat_old.csv'))
    elif x== 'MeatPig':
        final,prophet,model_neural,acc_list=pre('carbonemission/indiaDataset/meat_pig_old.csv','carbonemission/combined_data.csv',"carbonemission/combined_data_arima.csv","carbonemission/combined_data_neural_prophet.csv","Meat Pig",years)
        final1=pd.DataFrame(preprocessing('carbonemission/indiaDataset/meat_pig_old.csv'))
    elif x== 'MeatSheep':
       final,prophet,model_neural,acc_list=pre('carbonemission/indiaDataset/meat_sheep_old.csv','carbonemission/combined_data.csv',"carbonemission/combined_data_arima.csv","carbonemission/combined_data_neural_prophet.csv","Meat Sheep",years)
       final1=pd.DataFrame( preprocessing('carbonemission/indiaDataset/meat_sheep_old.csv'))
    elif x== 'MilkCow':
        final,prophet,model_neural,acc_list=pre('carbonemission/indiaDataset/milk_whole__fresh_cow_old.csv','carbonemission/combined_data.csv',"carbonemission/combined_data_arima.csv","carbonemission/combined_data_neural_prophet.csv","Milk Whole Fresh Cow",years)
        final1=pd.DataFrame(preprocessing('carbonemission/indiaDataset/milk_whole__fresh_cow_old.csv'))
    elif x== 'MilkBuffalo':
        final,prophet,model_neural,acc_list=pre('carbonemission/indiaDataset/milk_whole_fresh_buffalo_old.csv','carbonemission/combined_data.csv',"carbonemission/combined_data_arima.csv","carbonemission/combined_data_neural_prophet.csv","Milk Whole Fresh Buffalo",years)
        final1=pd.DataFrame(preprocessing('carbonemission/indiaDataset/milk_whole_fresh_buffalo_old.csv'))
    else:
        final,prophet,model_neural,acc_list=pre('carbonemission/indiaDataset/eggs_hen_in_shell_old.csv','carbonemission/combined_data.csv',"carbonemission/combined_data_arima.csv","carbonemission/combined_data_neural_prophet.csv","Egg hen in shell",years)
        final1=pd.DataFrame(preprocessing('carbonemission/indiaDataset/eggs_hen_in_shell_old.csv'))
    final=pd.DataFrame(final)
    prophet=pd.DataFrame(prophet)
    model_neural=pd.DataFrame(model_neural)
    return final,final1,prophet,model_neural,acc_list
    

def getVisiualization(start,end):

    combined_data=pd.read_csv('carbonemission/combined_data.csv')
    combined_data=combined_data.loc[(combined_data['Year']>=start)&(combined_data['Year']<=end)]
    fig1 = px.bar(combined_data, x="Year", y="Emission Intensity", color='Item',width=930, height=580)
    plot_div1 = plot(fig1, output_type='div', include_plotlyjs=False)
    fig2 = px.line(combined_data, x='Year', y='Emission Intensity', color='Item',width=930, height=580)
    plot_div2 = plot(fig2, output_type='div', include_plotlyjs=False)
    x = combined_data['Year']
    y = combined_data['Item']
    z = combined_data['Emission Intensity']
    fig3 = go.Figure(data=go.Heatmap(z=z, x=x, y=y, colorscale='rdylgn'))
    fig3.update_layout(title='Carbon Emission in Meat Industry', xaxis_nticks=36,height=560,width=930)
    plot_div3 = plot(fig3, output_type='div', include_plotlyjs=False)

    fig4 = px.scatter(combined_data, y="Emission Intensity", x="Year", color="Item",height=580,width=930)
    fig4.update_traces(marker_size=10)
    plot_div4 = plot(fig4, output_type='div', include_plotlyjs=False)

    return plot_div1,plot_div2,plot_div3,plot_div4
    
   

def visualize(request):
    start =(request.GET['start_date'])
    end =(request.GET['end_date'])
    
    fig1,fig2,fig3,fig4=getVisiualization(start,end)

    context={'fig1':fig1,'fig2':fig2,'fig3':fig3,'fig4':fig4}
    return render(request,"visualize.html",context)
    






def result(request):
    years = int(request.GET['years'])
    items = request.GET['typeOfItem']
    # result,actual =  pd.DataFrame(getPredictions(items,years))
    result,actual,prophet,model_neural,acc =  getPredictions(items,years)
    result=pd.DataFrame(result)
    actual=pd.DataFrame(actual)
    #img = result.plot(x='ds',y='yhat')
    img = plot({'data':[Scatter(x=result['Year'], y=result['Emission Intensity'],mode='lines+markers', name='Predicted Data', opacity=0.8, marker_color='blue'),Scatter(x=actual['Year'], y=actual['Value'],mode='lines+markers', name='Actual Data', opacity=0.8, marker_color='red')],'layout': {'xaxis': {'title': 'Year'}, 'yaxis': {'title': 'Carbon Emission'},'height':600}}, output_type='div')
    # plot({
    # 'data': [Scatter(...)],
    # 'layout': {'title': 'title', 'xaxis': {'title': 'xaxis_title'}, 'yaxis': {'title': 'yaxis_title'}}
    # }, output_type='div')
    # result=result.to_html(classes='table table-striped')
    # #print(result)
    # return render(request, 'result.html', {'result': result,'img':img})
    
    result['Year'] = result['Year'].dt.strftime('%Y-%m-%d')
    result['actualdata5']=actual['Value']
    result.rename(columns = {'Emission Intensity':'Value'},inplace=True)
    json_records = result.reset_index().to_json(orient ='records')
    data = []
    data = json.loads(json_records)
    context = {'d': data,'img':img,'items':items}
    return render(request, 'result.html', context)
    
    #return HttpResponse(result.to_html())
    # return render(request,'result.html',{'result':result})  


def com_res(request):
    years = int(request.GET['years'])
    items = request.GET['typeOfItem']
    # result,actual =  pd.DataFrame(getPredictions(items,years))
    result,actual,prophet,model_neural,acc_list =  getPredictions(items,years)
    result=pd.DataFrame(result)
    actual=pd.DataFrame(actual)
    prophet=pd.DataFrame(prophet)
    model_neural=pd.DataFrame(model_neural)
    img = plot({'data':[Scatter(x=result['Year'], y=result['Emission Intensity'],mode='lines+markers', name='Arima Predicted Data', opacity=0.8, marker_color='black'),Scatter(x=actual['Year'], y=actual['Value'],mode='lines+markers', name='Actual Data', opacity=0.8, marker_color='red'),Scatter(x=model_neural['ds'], y=model_neural['yhat1'],mode='lines+markers', name='Neural', opacity=0.8, marker_color='yellow'),Scatter(x=prophet['Year'], y=prophet['Emission Intensity'],mode='lines+markers', name='Prophet Predicted Data', opacity=0.8, marker_color='blue')],'layout': {'xaxis': {'title': 'Year'}, 'yaxis': {'title': 'Emission Intensity'},'margin':{'t':15,'b':15},'height':650}}, output_type='div')
    
    result['Year'] = result['Year'].dt.strftime('%Y-%m-%d')
    result.rename(columns = {'Emission Intensity':'Value'},inplace=True)
    result['actualdata5']=actual['Value']
    result['prophet']=prophet['Emission Intensity']
    result['neuralprophet']=model_neural['yhat1']
    json_records = result.reset_index().to_json(orient ='records')
    data = []
    data = json.loads(json_records)
    context = {'d': data,'img':img,'items':items,'acc_list':acc_list}
    return render(request, 'com_res.html', context)

# ARIMA visualize
def A_getVisiualization(start,end):

    combined_data=pd.read_csv('carbonemission/combined_data_arima.csv')
    combined_data=combined_data.loc[(combined_data['Year']>=start)&(combined_data['Year']<=end)]
    fig1 = px.bar(combined_data, x="Year", y="Emission Intensity", color='Item',width=930, height=580)
    plot_div1 = plot(fig1, output_type='div', include_plotlyjs=False)
    fig2 = px.line(combined_data, x='Year', y='Emission Intensity', color='Item',width=930, height=580)
    plot_div2 = plot(fig2, output_type='div', include_plotlyjs=False)
    x = combined_data['Year']
    y = combined_data['Item']
    z = combined_data['Emission Intensity']
    fig3 = go.Figure(data=go.Heatmap(z=z, x=x, y=y, colorscale='rdylgn'))
    fig3.update_layout(title='Carbon Emission in Meat Industry', xaxis_nticks=36,height=560,width=930)
    plot_div3 = plot(fig3, output_type='div', include_plotlyjs=False)

    fig4 = px.scatter(combined_data, y="Emission Intensity", x="Year", color="Item",height=580,width=930)
    fig4.update_traces(marker_size=10)
    plot_div4 = plot(fig4, output_type='div', include_plotlyjs=False)

    return plot_div1,plot_div2,plot_div3,plot_div4
    
   

def A_visualize(request):
    start =(request.GET['start_date'])
    end =(request.GET['end_date'])
    
    fig1,fig2,fig3,fig4=A_getVisiualization(start,end)

    context={'fig1':fig1,'fig2':fig2,'fig3':fig3,'fig4':fig4}
    return render(request,"A_visualize.html",context)

# NeuralProphet visualize
def N_getVisiualization(start,end):

    combined_data=pd.read_csv('carbonemission/combined_data_neural_prophet.csv')
    combined_data.rename(columns = {'ds' : 'Year', 'yhat1' : 'Emission Intensity'}, inplace = True)
    combined_data=combined_data.loc[(combined_data['Year']>=start)&(combined_data['Year']<=end)]
    fig1 = px.bar(combined_data, x="Year", y="Emission Intensity", color='Item',width=930, height=580)
    plot_div1 = plot(fig1, output_type='div', include_plotlyjs=False)
    fig2 = px.line(combined_data, x='Year', y='Emission Intensity', color='Item',width=930, height=580)
    plot_div2 = plot(fig2, output_type='div', include_plotlyjs=False)
    x = combined_data['Year']
    y = combined_data['Item']
    z = combined_data['Emission Intensity']
    fig3 = go.Figure(data=go.Heatmap(z=z, x=x, y=y, colorscale='rdylgn'))
    fig3.update_layout(title='Carbon Emission in Meat Industry', xaxis_nticks=36,height=560,width=930)
    plot_div3 = plot(fig3, output_type='div', include_plotlyjs=False)

    fig4 = px.scatter(combined_data, y="Emission Intensity", x="Year", color="Item",height=580,width=930)
    fig4.update_traces(marker_size=10)
    plot_div4 = plot(fig4, output_type='div', include_plotlyjs=False)

    return plot_div1,plot_div2,plot_div3,plot_div4
    
    
   

def N_visualize(request):
    start =(request.GET['start_date'])
    end =(request.GET['end_date'])
    
    fig1,fig2,fig3,fig4=N_getVisiualization(start,end)

    context={'fig1':fig1,'fig2':fig2,'fig3':fig3,'fig4':fig4}
    return render(request,"N_visualize.html",context)
