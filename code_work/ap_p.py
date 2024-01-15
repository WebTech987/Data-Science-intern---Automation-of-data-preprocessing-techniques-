from sanic import Sanic, response
from sanic.response import html
from sanic.request import Request
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

import pandas as pd
import regex as re
import numpy as np
import requests
import json
import math

#import pickle5 as pickle
app = Sanic(__name__)

#definig the global variables
df = None  #The variable we will perform all the operations
df_original = None #variable to store the data and keep it till the end we might need 
df_1 = None #Variable that stores the updated data 
finalCols = {} #Declaring the dictionary which will store the columns to perform data processing

# Route for the home page
@app.route('/')
async def home(request):
    return html(open('index.html').read())


# ------- Data Description ------------------------------------------------


# Route to handle the file upload and showing the data description
@app.route('/upload', methods=['POST'])
async def upload(request: Request):
    global df
    global df_original
    global df_1
    uploaded_file = request.files.get('file')

    if uploaded_file:
        # Save the uploaded file to a temporary location
        file_path = f'tmp/{uploaded_file.name}'
        with open(file_path, 'wb') as file:
            file.write(uploaded_file.body)

            
        # Convert file to CSV and extract data, storing original data
        df = pd.read_csv(file_path)
        df_original = df.copy()
        df_original.columns = df_original.columns.str.lower()
        df.columns = df.columns.str.lower()
        df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)
        df_original = df_original.applymap(lambda y: y.lower() if isinstance(y, str) else y)

        # Perform data analysis and generate summary
        rows, columns = df.shape
        column_summaries = []
        for column in df.columns:
            column_summary = {
                'column_name': column,
                'unique_values': df[column].nunique(),
                'value_counts': df[column].value_counts().to_dict(),
                'null_values': df[column].isnull().sum()
            }
            column_summaries.append(column_summary)
      
        rows = int(rows)  # Convert rows to integer
        columns = int(columns)  # Convert columns to integer
        #storing the column names
        column_namess = []
        i = 1
        for items in df.columns:
            column_name = {
                str(i) : items, 
            }
            i += 1
            column_namess.append(column_name)

        # Convert int64 values in column_summaries to integers
        column_summaries = [
            {
                'column_name': summary['column_name'],
                'unique_values': int(summary['unique_values']),
                'value_counts': {str(key): int(value) for key, value in summary['value_counts'].items()},
                'null_values': int(summary['null_values'])
            }
            for summary in column_summaries
        ]
        # Render the result page with the summary data
        return html(open('result.html').read().replace('{rows}', str(rows)).replace('{columns}', str(columns)).replace('{column_summaries}', json.dumps(column_summaries)).replace('{df_json}', json.dumps(column_namess)))

    return html('No file was uploaded.')


#--------Columns selections and route to the we-page to show the selected columns-----

# API takes the selected columns from user and then proces the data to next step
@app.route('/submit', methods=['POST'])
async def submit(request: Request):
    global df
    global finalCols
    global df_1
    selected_columns = request.json.get('columns')
    no_rows, no_cols = df.shape

    #Storing the selected columsn in a global variable for further use
    i = 1
    for items in selected_columns:
        if items not in finalCols:
            finalCols[items] = str(i)
            i += 1
    
    selected_columns = []
    for items in finalCols.keys():
        selected_columns += [items]
    # Use the selected columns for further processing
    # Example: Access the DataFrame from the previous API function
    # df = request['df']
    #Filtering the selected columns that needs to go ahead
#    df = df.filter(items = selected_columns) #modifying the data
    print("Selected columns:", selected_columns)
    print("New shape of the dataframe becomes", df.shape)
    print("the columns stored as globalls are: ", finalCols)
    
    #Perform The Data Cleaning
    # Perform desired operations with the selected columns
#    pickled_model = pickle.load(open('state_district.pkl', 'rb'))
#    print("This is getting executed")

    # Return the response
    return html(open('result.html').read().replace('{number_l}', str(len(selected_columns))))

#------------Graph plotting -----

#Api to return or print the graph
@app.route("/graph", methods=["GET"])
async def generate_graph(request):
    global finalCols
    global df
    global df_1
    # Generate the graph using matplotlib

    # Getting columns with null values
    columns_with_nulls = df.columns[df.isnull().any()].tolist()

    # Calculate null value frequencies
    null_frequencies = df[columns_with_nulls].isnull().sum()

    # Plot the null value frequencies
    plt.bar(null_frequencies.index, null_frequencies.values, width = 0.3, color = "g")
    plt.xlabel('Columns')
    plt.ylabel('Null Values Frequency')
    plt.title('Data Completeness')
    plt.xticks(rotation = 30)
    # Save the graph as an image
    graph_path = "graph.png"
    plt.savefig(graph_path)
    plt.close()

    # Read the image file
    with open(graph_path, "rb") as f:
        graph_data = f.read()

    # Delete the image file
    import os
    os.remove(graph_path)

    # Send the graph as a response
    headers = {'Content-Type': 'image/png'}
    return response.raw(graph_data, headers=headers)



#-------------------Rendering a new page reultt---------------

#--------------------- DATA CLEANING ---------------------

#-------------------API to reach to the data cleaning page from the server
@app.route('/cleaning', methods=['GET'])
async def open_html_file(request):
    return await response.file('resultt.html')

#---------API to correct the sensitive column values such as state, district or pincode-----
@app.route('/correctState', methods = ['POST'])
async def correctState(request: Request):
    global df
    global finalCols
    global df_1


    df.columns = df.columns.str.lower()
    df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)

    if 'states' in df.columns:
        df['state'] = df['states']
        df = df.drop(columns = ['states'], axis = 0)
    if 'pin' in df.columns:
        df['pincode'] = df['pin']
        df = df.drop(columns = ['pin'], axis = 0)
    if 'districts' in df.columns:
        df['district'] = df['districts']
        df = df.drop(columns = ['districts'], axis = 0)
    if 'distt' in df.columns:
        df['district'] = df['distt']
        df = df.drop(columns = ['distt'], axis = 0)

    if 'pincode' not in df.columns:
        df['pincode'] = np.nan
    if 'state' not in df.columns:
        df['state'] = np.nan
    if 'district' not in df.columns:
        df['district'] = np.nan
    if 'address' not in df.columns:
        df['address'] = np.nan
    if 'latitude' and 'longitude' not in df.columns:
        df['latitude'] = np.nan
        df['longitude'] = np.nan
    

    def state_district(data):
        
        #First storing all the states and union terrioterie present in india in the lowercase strings
        all_states_un = {'andaman nicobar', 'andhra pradesh', 'arunachal pradesh', 'assam', 'bihar', 'chandigarh', 'chhattisgarh', 'dadra and nagar haveli and daman and diu', 'delhi', 'goa', 'gujarat', 'haryana', 'himachal pradesh', 'jammu kashmir', 'jharkhand', 'karnataka', 'kerala', 'ladakh', 'lakshadweep', 'madhya pradesh', 'maharashtra', 'manipur', 'meghalaya', 'mizoram', 'nagaland', 'odisha', 'puducherry', 'punjab', 'rajasthan', 'sikkim', 'tamil nadu', 'telangana', 'tripura', 'uttar pradesh', 'uttarakhand', 'west bengal'}

        #Containing all the states and their abbreviation along with union terrioteries
        states_dict = {'andaman nicobar':'an','andhra pradesh': 'ap', 'arunachal pradesh': 'ar', 'assam': 'as', 'bihar': 'br', 'chhattisgarh': 'cg', 'goa': 'ga', 'gujarat': 'gj', 'haryana': 'hr', 'himachal pradesh': 'hp', 'jammu and kashmir': 'jk', 'jammu kashmir': 'jk', 'jharkhand': 'jh', 'karnataka': 'ka', 'kerala': 'kl', 'madhya pradesh': 'mp', 'maharashtra': 'mh', 'manipur': 'mn', 'meghalaya': 'ml', 'mizoram': 'mz', 'nagaland': 'nl', 'orissa': 'or', 'odisha': 'or', 'punjab': 'pb', 'rajasthan': 'rj', 'sikkim': 'sk', 'tamil nadu': 'tn', 'tripura': 'tr', 'uttarakhand': 'uk', 'uttar pradesh': 'up', 'west bengal': 'wb', 'andaman and nicobar islands': 'an', 'chandigarh': 'ch', 'dadra and nagar haveli': 'dh', 'daman and diu': 'dd', 'delhi': 'dl', 'lakshadweep': 'ld', 'pondicherry': 'py', 'ponducherry' : 'py', 'dadra and nagar haveli and daman and diu' : 'dhdd'}
        #First storing all the districts present in india in the lowercase strings
        all_districts = {'adilabad', 'agar malwa', 'agra', 'ahmedabad', 'ahmednagar', 'aizawl', 'ajmer', 'akola', 'alappuzha', 'aligarh', 'alipurduar', 'alirajpur', 'alluri sitarama raju', 'almora', 'alwar', 'ambala', 'ambedkar nagar', 'amethi', 'amravati', 'amreli', 'amritsar', 'amroha', 'anakapalli', 'anand', 'anantapur', 'anantnag', 'angul', 'anjaw', 'annamaya', 'anuppur', 'araria', 'aravalli', 'ariyalur', 'arwal', 'ashoknagar', 'auraiya', 'aurangabad', 'ayodhya', 'azamgarh', 'bagalkot', 'bageshwar', 'baghpat', 'bahraich', 'baksa', 'balaghat', 'balangir', 'balasore', 'ballia', 'balod', 'baloda bazar', 'balrampur', 'banaskantha', 'banda', 'bandipora', 'bangalore rural', 'bangalore urban', 'banka', 'bankura', 'banswara', 'bapatla', 'barabanki', 'baramulla', 'baran', 'bareilly', 'bargarh', 'barmer', 'barnala', 'barpeta', 'barwani', 'bastar', 'basti', 'bathinda', 'beed', 'begusarai', 'belgaum', 'bellary', 'bemetara', 'betul', 'bhadohi', 'bhadradri kothagudem', 'bhadrak', 'bhagalpur', 'bhandara', 'bharatpur', 'bharuch', 'bhavnagar', 'bhilwara', 'bhind', 'bhiwani', 'bhojpur', 'bhopal', 'bidar', 'bijapur', 'bijnor', 'bikaner', 'bilaspur', 'birbhum', 'bishnupur', 'bokaro', 'bongaigaon', 'botad', 'boudh', 'budaun', 'budgam', 'bulandshahr', 'buldhana', 'bundi', 'burhanpur', 'buxar', 'cachar', 'central delhi', 'central siang', 'chachaura', 'chamarajanagar', 'chamba', 'chamoli', 'champawat', 'champhai', 'chandauli', 'chandel', 'chandigarh', 'chandrapur', 'changlang', 'charaideo', 'charkhi dadri', 'chatra', 'chengalpattu', 'chennai', 'chhatarpur', 'chhindwara', 'chhota udaipur', 'chikkaballapur', 'chikkamagaluru', 'chirang', 'chitradurga', 'chitrakoot', 'chittoor', 'chittorgarh', 'chumukedima', 'churachandpur', 'churu', 'coimbatore', 'cooch behar', 'cuddalore', 'cuttack', 'dadra and nagar haveli', 'dahod', 'dakshin dinajpur', 'dakshina kannada', 'daman', 'damoh', 'dang', 'dantewada', 'darbhanga', 'darjeeling', 'darrang', 'datia', 'dausa', 'davanagere', 'debagarh', 'dehradun', 'deoghar', 'deoria', 'devbhoomi dwarka', 'dewas', 'dhalai', 'dhamtari', 'dhanbad', 'dhar', 'dharmapuri', 'dharwad', 'dhemaji', 'dhenkanal', 'dholpur', 'dhubri', 'dhule', 'dibang valley', 'dibrugarh', 'dima hasao', 'dimapur', 'dindigul', 'dindori', 'diu', 'doda', 'dumka', 'dungarpur', 'durg', 'east champaran', 'east delhi', 'east garo hills', 'east godavari', 'east jaintia hills', 'east kameng', 'east khasi hills', 'east siang', 'east sikkim', 'east singhbhum', 'eluru', 'ernakulam', 'erode', 'etah', 'etawah', 'faridabad', 'faridkot', 'farrukhabad', 'fatehabad', 'fatehgarh sahib', 'fatehpur', 'fazilka', 'firozabad', 'firozpur', 'gadag', 'gadchiroli', 'gajapati', 'ganderbal', 'gandhinagar', 'ganjam', 'garhwa', 'gariaband', 'gaurela pendra marwahi', 'gautam buddha nagar', 'gaya', 'ghaziabad', 'ghazipur', 'gir somnath', 'giridih', 'goalpara', 'godda', 'golaghat', 'gomati', 'gonda', 'gondia', 'gopalganj', 'gorakhpur', 'gulbarga', 'gumla', 'guna', 'guntur', 'gurdaspur', 'gurugram', 'gwalior', 'hailakandi', 'hamirpur', 'hanamkonda', 'hanumangarh', 'hapur', 'harda', 'hardoi', 'haridwar', 'hassan', 'hathras', 'haveri', 'hazaribagh', 'hingoli', 'hisar', 'hnahthial', 'hooghly', 'hoshangabad', 'hoshiarpur', 'howrah', 'hyderabad', 'idukki', 'imphal east', 'imphal west', 'indore', 'jabalpur', 'jagatsinghpur', 'jagtial', 'jaipur', 'jaisalmer', 'jajpur', 'jalandhar', 'jalaun', 'jalgaon', 'jalna', 'jalore', 'jalpaiguri', 'jammu', 'jamnagar', 'jamtara', 'jamui', 'jangaon', 'janjgir champa', 'jashpur', 'jaunpur', 'jayashankar', 'jehanabad', 'jhabua', 'jhajjar', 'jhalawar', 'jhansi', 'jhargram', 'jharsuguda', 'jhunjhunu', 'jind', 'jiribam', 'jodhpur', 'jogulamba', 'jorhat', 'junagadh', 'kabirdham', 'kadapa', 'kaimur', 'kaithal', 'kakching', 'kakinada ', 'kalahandi', 'kalimpong', 'kallakurichi', 'kamareddy', 'kamjong', 'kamle', 'kamrup', 'kamrup metropolitan', 'kanchipuram', 'kandhamal', 'kangpokpi', 'kangra', 'kanker', 'kannauj', 'kannur', 'kanpur dehat', 'kanpur nagar', 'kanyakumari', 'kapurthala', 'karaikal', 'karauli', 'karbi anglong', 'kargil', 'karimganj', 'karimnagar', 'karnal', 'karur', 'kasaragod', 'kasganj', 'kathua', 'katihar', 'katni', 'kaushambi', 'kendrapara', 'kendujhar', 'khagaria', 'khairagarh', 'khammam', 'khandwa', 'khargone', 'khawzawl', 'kheda', 'kheri', 'khordha', 'khowai', 'khunti', 'kinnaur', 'kiphire', 'kishanganj', 'kishtwar', 'kodagu', 'koderma', 'kohima', 'kokrajhar', 'kolar', 'kolasib', 'kolhapur', 'kolkata', 'kollam', 'komaram bheem', 'konaseema', 'kondagaon', 'koppal', 'koraput', 'korba', 'koriya', 'kota', 'kottayam', 'kozhikode', 'kra daadi', 'krishna', 'krishnagiri', 'kulgam', 'kullu', 'kupwara', 'kurnool', 'kurukshetra', 'kurung kumey', 'kushinagar', 'kutch', 'lahaul spiti', 'lakhimpur', 'lakhisarai', 'lakshadweep', 'lalitpur', 'latehar', 'latur', 'lawngtlai', 'leh', 'lepa rada', 'lohardaga', 'lohit', 'longding', 'longleng', 'lower dibang valley', 'lower siang', 'lower subansiri', 'lucknow', 'ludhiana', 'lunglei', 'madhepura', 'madhubani', 'madurai', 'mahabubabad', 'maharajganj', 'mahasamund', 'mahbubnagar', 'mahe', 'mahendragarh', 'mahisagar', 'mahoba', 'maihar', 'mainpuri', 'mairang', 'majuli', 'malappuram', 'malda', 'malerkotla', 'malkangiri', 'mamit', 'mancherial', 'mandi', 'mandla', 'mandsaur', 'mandya', 'manendragarh', 'mansa', 'manyam', 'mathura', 'mau', 'mayiladuthurai ', 'mayurbhanj', 'medak', 'medchal', 'meerut', 'mehsana', 'mewat', 'mirzapur', 'moga', 'mohali', 'mohla manpur', 'mokokchung', 'mon', 'moradabad', 'morbi', 'morena', 'morigaon', 'muktsar', 'mulugu', 'mumbai city', 'mumbai suburban', 'mungeli', 'munger', 'murshidabad', 'muzaffarnagar', 'muzaffarpur', 'mysore', 'n t rama rao', 'nabarangpur', 'nadia', 'nagaon', 'nagapattinam', 'nagarkurnool', 'nagaur', 'nagda', 'nagpur', 'nainital', 'nalanda', 'nalbari', 'nalgonda', 'namakkal', 'namsai', 'nanded', 'nandurbar', 'nandyal', 'narayanpet', 'narayanpur', 'narmada', 'narsinghpur', 'nashik', 'navsari', 'nawada', 'nayagarh', 'neemuch', 'nellore', 'new delhi', 'nicobar', 'nilgiris', 'nirmal', 'niuland', 'niwari', 'nizamabad', 'noklak', 'noney', 'north 24 parganas', 'north delhi', 'north east delhi', 'north garo hills', 'north goa', 'north middle andaman', 'north sikkim', 'north tripura', 'north west delhi', 'nuapada', 'osmanabad', 'pakke kessang', 'pakur', 'pakyong', 'palakkad', 'palamu', 'palghar', 'pali', 'palnadu', 'palwal', 'panchkula', 'panchmahal', 'panipat', 'panna', 'papum pare', 'parbhani', 'paschim bardhaman', 'paschim medinipur', 'patan', 'pathanamthitta', 'pathankot', 'patiala', 'patna', 'pauri', 'peddapalli', 'perambalur', 'peren', 'phek', 'pherzawl', 'pilibhit', 'pithoragarh', 'poonch', 'porbandar', 'prakasam', 'pratapgarh', 'prayagraj', 'puducherry', 'pudukkottai', 'pulwama', 'pune', 'purba bardhaman', 'purba medinipur', 'puri', 'purnia', 'purulia', 'raebareli', 'raichur', 'raigad', 'raigarh', 'raipur', 'raisen', 'rajanna sircilla', 'rajgarh', 'rajkot', 'rajnandgaon', 'rajouri', 'rajsamand', 'ramanagara', 'ramanathapuram', 'ramban', 'ramgarh', 'rampur', 'ranchi', 'ranga reddy', 'ranipet', 'ratlam', 'ratnagiri', 'rayagada', 'reasi', 'rewa', 'rewari', 'ri bhoi', 'rohtak', 'rohtas', 'rudraprayag', 'rupnagar', 'sabarkantha', 'sagar', 'saharanpur', 'saharsa', 'sahebganj', 'saiha', 'saitual', 'sakti', 'salem', 'samastipur', 'samba', 'sambalpur', 'sambhal', 'sangareddy', 'sangli', 'sangrur', 'sant kabir nagar', 'saran', 'sarangarh bilaigarh', 'satara', 'satna', 'sawai madhopur', 'sehore', 'senapati', 'seoni', 'sepahijala', 'seraikela kharsawan', 'serchhip', 'shahdara', 'shahdol', 'shaheed bhagat singh nagar', 'shahjahanpur', 'shajapur', 'shamator', 'shamli', 'sheikhpura', 'sheohar', 'sheopur', 'shi yomi', 'shimla', 'shimoga', 'shivpuri', 'shopian', 'shravasti', 'siddharthnagar', 'siddipet', 'sidhi', 'sikar', 'simdega', 'sindhudurg', 'singrauli', 'sirmaur', 'sirohi', 'sirsa', 'sitamarhi', 'sitapur', 'sivaganga', 'sivasagar', 'siwan', 'solan', 'solapur', 'sonbhadra', 'sonipat', 'sonitpur', 'soreng', 'south 24 parganas', 'south andaman', 'south delhi', 'south east delhi', 'south garo hills', 'south goa', 'south salmara-mankachar', 'south sikkim', 'south tripura', 'south west delhi', 'south west garo hills', 'south west khasi hills', 'sri balaji', 'sri ganganagar', 'sri satya sai', 'srikakulam', 'srinagar', 'subarnapur', 'sukma', 'sultanpur', 'sundergarh', 'supaul', 'surajpur', 'surat', 'surendranagar', 'surguja', 'suryapet', 'tamenglong', 'tapi', 'tarn taran', 'tawang', 'tehri', 'tengnoupal', 'tenkasi', 'thane', 'thanjavur', 'theni', 'thiruvananthapuram', 'thoothukudi', 'thoubal', 'thrissur', 'tikamgarh', 'tinsukia', 'tirap', 'tiruchirappalli', 'tirunelveli', 'tirupattur', 'tiruppur', 'tiruvallur', 'tiruvannamalai', 'tiruvarur', 'tonk', 'tseminyu', 'tuensang', 'tumkur', 'udaipur', 'udalguri', 'udham singh nagar', 'udhampur', 'udupi', 'ujjain', 'ukhrul', 'umaria', 'una', 'unakoti', 'unnao', 'upper siang', 'upper subansiri', 'uttar dinajpur', 'uttara kannada', 'uttarkashi', 'vadodara', 'vaishali', 'valsad', 'varanasi', 'vellore', 'vidisha', 'vijayanagara', 'vijayapura ', 'vikarabad', 'viluppuram', 'virudhunagar', 'visakhapatnam', 'vizianagaram', 'wanaparthy', 'warangal', 'wardha', 'washim', 'wayanad', 'west champaran', 'west delhi', 'west garo hills', 'west godavari', 'west jaintia hills', 'west kameng', 'west karbi anglong', 'west khasi hills', 'west siang', 'west sikkim', 'west singhbhum', 'west tripura', 'wokha', 'yadadri bhuvanagiri', 'yadgir', 'yamunanagar', 'yanam', 'yavatmal', 'zunheboto'}
        
        #Converting all the string values present in the df into lowercase
        data = data.applymap(lambda x: x.lower() if isinstance(x, str) else x)

        #converting the columns names into lowercase
        data.columns = data.columns.str.lower()
        
        #storing data columns
        data_cols = set()
        for items in data.columns:
            data_cols.add(items)
        
        #Making the state value None if wrong input is present in the name of a state
        i = 0
        for item in data['state']:
            if item not in states_dict:
                data['state'][i] = None
            i += 1
        
        if 'district' in data_cols:
            #Making the state value None if wrong input is present in the name of a state
            i = 0
            for item in data['district']:
                if item not in all_districts:
                    data['district'][i] = None
                i += 1
        if 'pincode' in data_cols:
            #Making the pincode value None if wrong input is present in the name of a state
            i = 0
            for item in data['pincode']:
                if len(item) != 6:
                    data['pincode'][i] = None
                i += 1
        return data

    #Function to check whether any string is present in a numeric type column
    def clean_column(data, column_name):
        # Calculate the percentage of numeric values in the column
        numeric_percentage = (data[column_name].apply(lambda x: isinstance(x, (int, float))).sum() / len(data)) * 100
        char_percentage = (data[column_name].apply(lambda x: isinstance(x, str)).sum() / len(data)) * 100
        # Check if the percentage of numeric values is above 90
        if numeric_percentage > 92:
            # Set non-numeric values to None
            data.loc[~data[column_name].apply(lambda x: isinstance(x, (int, float))), column_name] = None

        if char_percentage > 92 :
            data.loc[~data[column_name].apply(lambda x: isinstance(x, str)), column_name] = None

        return data


    def shape_address(data):
        """
        Shape the "Address" column values in a pandas DataFrame into a format that can be used to find the latitude and longitude values.
        """
        # define a function to clean and shape the address string
        def clean_address(address):
            # replace any multiple spaces with a single space
            address = re.sub('\s+', ' ', str(address))
            # remove any leading or trailing spaces
            address = address.strip()
            # remove any commas
            address = address.replace(',', '').replace('/', '').replace('%', '').replace('"', '').replace('?', '').replace('|', '').replace('*', '').replace('&', '')
            # replace any double quotes with single quotes
            address = address.replace('"', "'")
            # return the cleaned address string
            return address
        
        # iterate over the rows of the DataFrame
        for index, row in data.iterrows():
            # get the address from the "Address" column
            address = row["address"]
            # if the address is missing, skip this row
            if pd.isna(address):
                continue
            # clean and shape the address string using the clean_address function
            address = clean_address(address)
            # update the DataFrame with the new value
            data.at[index, "address"] = address
        
        # return the updated DataFrame
        return data

    # Calling the function for every column that user have selected to make the minority None based on variable type present
    for items in finalCols.keys():
        df = clean_column(df, items)

    if 'address' in df.columns:
        df = shape_address(df)
    
    #cleaning state, district, pincode  values in the df
    if 'state' in df.columns or 'district' in df.columns or 'pincode' in df.columns:
        df = state_district(df)

    json_data = df.to_json(orient='records')
    print("This is getting executed")
    return response.json(json_data)


#----------------------API to fill null values or perform imputaion
@app.route("/fill_null", methods=["GET"])
async def fill_null_values(request:Request):
    # Fill null values in the DataFrame
    global df
    global df_1
    global finalCols
    print("fill null is getting executed")
    df.columns = df.columns.str.lower()
    df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)
    print("x, y, z", df.columns)
    
#-----------code from the correctState API :

    if 'states' in df.columns:
        df['state'] = df['states']
        df = df.drop(columns='states', axis=1)
        df = df.copy()
    if 'pin' in df.columns:
        df['pincode'] = df['pin']
        df = df.drop(columns = 'pin', axis = 1)
        df = df.copy()
    if 'districts' in df.columns:
        df['district'] = df['districts']
        df = df.drop(columns = 'districts', axis = 1)
        df = df.copy()
    if 'distt' in df.columns:
        df['district'] = df['distt']
        df = df.drop(columns = 'distt', axis = 1)
        df = df.copy()

    if 'pincode' not in df.columns:
        df['pincode'] = np.nan
        df = df.copy()
    if 'state' not in df.columns:
        df['state'] = np.nan
        df = df.copy()
    if 'district' not in df.columns:
        df['district'] = np.nan
        df = df.copy()
    if 'address' not in df.columns:
        df['address'] = np.nan
        df = df.copy()
    if 'latitude' not in df.columns and 'longitude'  not in df.columns:
        df['latitude'] = np.nan
        df['longitude'] = np.nan
        df = df.copy()

    data1 = df.copy()

    def state_district(data_set):

        data = data_set.copy()
        #First storing all the states and union terrioterie present in india in the lowercase strings
        all_states_un = {'andaman nicobar', 'andhra pradesh', 'arunachal pradesh', 'assam', 'bihar', 'chandigarh', 'chhattisgarh', 'dadra and nagar haveli and daman and diu', 'delhi', 'goa', 'gujarat', 'haryana', 'himachal pradesh', 'jammu kashmir', 'jharkhand', 'karnataka', 'kerala', 'ladakh', 'lakshadweep', 'madhya pradesh', 'maharashtra', 'manipur', 'meghalaya', 'mizoram', 'nagaland', 'odisha', 'puducherry', 'punjab', 'rajasthan', 'sikkim', 'tamil nadu', 'telangana', 'tripura', 'uttar pradesh', 'uttarakhand', 'west bengal'}

        #Containing all the states and their abbreviation along with union terrioteries
        states_dict = {'andaman nicobar':'an','andhra pradesh': 'ap', 'arunachal pradesh': 'ar', 'assam': 'as', 'bihar': 'br', 'chhattisgarh': 'cg', 'goa': 'ga', 'gujarat': 'gj', 'haryana': 'hr', 'himachal pradesh': 'hp', 'jammu and kashmir': 'jk', 'jammu kashmir': 'jk', 'jharkhand': 'jh', 'karnataka': 'ka', 'kerala': 'kl', 'madhya pradesh': 'mp', 'maharashtra': 'mh', 'manipur': 'mn', 'meghalaya': 'ml', 'mizoram': 'mz', 'nagaland': 'nl', 'orissa': 'or', 'odisha': 'or', 'punjab': 'pb', 'rajasthan': 'rj', 'sikkim': 'sk', 'tamil nadu': 'tn', 'tripura': 'tr', 'uttarakhand': 'uk', 'uttar pradesh': 'up', 'west bengal': 'wb', 'andaman and nicobar islands': 'an', 'chandigarh': 'ch', 'dadra and nagar haveli': 'dh', 'daman and diu': 'dd', 'delhi': 'dl', 'lakshadweep': 'ld', 'pondicherry': 'py', 'ponducherry' : 'py', 'dadra and nagar haveli and daman and diu' : 'dhdd'}
        #First storing all the districts present in india in the lowercase strings
        all_districts = {'adilabad', 'agar malwa', 'agra', 'ahmedabad', 'ahmednagar', 'aizawl', 'ajmer', 'akola', 'alappuzha', 'aligarh', 'alipurduar', 'alirajpur', 'alluri sitarama raju', 'almora', 'alwar', 'ambala', 'ambedkar nagar', 'amethi', 'amravati', 'amreli', 'amritsar', 'amroha', 'anakapalli', 'anand', 'anantapur', 'anantnag', 'angul', 'anjaw', 'annamaya', 'anuppur', 'araria', 'aravalli', 'ariyalur', 'arwal', 'ashoknagar', 'auraiya', 'aurangabad', 'ayodhya', 'azamgarh', 'bagalkot', 'bageshwar', 'baghpat', 'bahraich', 'baksa', 'balaghat', 'balangir', 'balasore', 'ballia', 'balod', 'baloda bazar', 'balrampur', 'banaskantha', 'banda', 'bandipora', 'bangalore rural', 'bangalore urban', 'banka', 'bankura', 'banswara', 'bapatla', 'barabanki', 'baramulla', 'baran', 'bareilly', 'bargarh', 'barmer', 'barnala', 'barpeta', 'barwani', 'bastar', 'basti', 'bathinda', 'beed', 'begusarai', 'belgaum', 'bellary', 'bemetara', 'betul', 'bhadohi', 'bhadradri kothagudem', 'bhadrak', 'bhagalpur', 'bhandara', 'bharatpur', 'bharuch', 'bhavnagar', 'bhilwara', 'bhind', 'bhiwani', 'bhojpur', 'bhopal', 'bidar', 'bijapur', 'bijnor', 'bikaner', 'bilaspur', 'birbhum', 'bishnupur', 'bokaro', 'bongaigaon', 'botad', 'boudh', 'budaun', 'budgam', 'bulandshahr', 'buldhana', 'bundi', 'burhanpur', 'buxar', 'cachar', 'central delhi', 'central siang', 'chachaura', 'chamarajanagar', 'chamba', 'chamoli', 'champawat', 'champhai', 'chandauli', 'chandel', 'chandigarh', 'chandrapur', 'changlang', 'charaideo', 'charkhi dadri', 'chatra', 'chengalpattu', 'chennai', 'chhatarpur', 'chhindwara', 'chhota udaipur', 'chikkaballapur', 'chikkamagaluru', 'chirang', 'chitradurga', 'chitrakoot', 'chittoor', 'chittorgarh', 'chumukedima', 'churachandpur', 'churu', 'coimbatore', 'cooch behar', 'cuddalore', 'cuttack', 'dadra and nagar haveli', 'dahod', 'dakshin dinajpur', 'dakshina kannada', 'daman', 'damoh', 'dang', 'dantewada', 'darbhanga', 'darjeeling', 'darrang', 'datia', 'dausa', 'davanagere', 'debagarh', 'dehradun', 'deoghar', 'deoria', 'devbhoomi dwarka', 'dewas', 'dhalai', 'dhamtari', 'dhanbad', 'dhar', 'dharmapuri', 'dharwad', 'dhemaji', 'dhenkanal', 'dholpur', 'dhubri', 'dhule', 'dibang valley', 'dibrugarh', 'dima hasao', 'dimapur', 'dindigul', 'dindori', 'diu', 'doda', 'dumka', 'dungarpur', 'durg', 'east champaran', 'east delhi', 'east garo hills', 'east godavari', 'east jaintia hills', 'east kameng', 'east khasi hills', 'east siang', 'east sikkim', 'east singhbhum', 'eluru', 'ernakulam', 'erode', 'etah', 'etawah', 'faridabad', 'faridkot', 'farrukhabad', 'fatehabad', 'fatehgarh sahib', 'fatehpur', 'fazilka', 'firozabad', 'firozpur', 'gadag', 'gadchiroli', 'gajapati', 'ganderbal', 'gandhinagar', 'ganjam', 'garhwa', 'gariaband', 'gaurela pendra marwahi', 'gautam buddha nagar', 'gaya', 'ghaziabad', 'ghazipur', 'gir somnath', 'giridih', 'goalpara', 'godda', 'golaghat', 'gomati', 'gonda', 'gondia', 'gopalganj', 'gorakhpur', 'gulbarga', 'gumla', 'guna', 'guntur', 'gurdaspur', 'gurugram', 'gwalior', 'hailakandi', 'hamirpur', 'hanamkonda', 'hanumangarh', 'hapur', 'harda', 'hardoi', 'haridwar', 'hassan', 'hathras', 'haveri', 'hazaribagh', 'hingoli', 'hisar', 'hnahthial', 'hooghly', 'hoshangabad', 'hoshiarpur', 'howrah', 'hyderabad', 'idukki', 'imphal east', 'imphal west', 'indore', 'jabalpur', 'jagatsinghpur', 'jagtial', 'jaipur', 'jaisalmer', 'jajpur', 'jalandhar', 'jalaun', 'jalgaon', 'jalna', 'jalore', 'jalpaiguri', 'jammu', 'jamnagar', 'jamtara', 'jamui', 'jangaon', 'janjgir champa', 'jashpur', 'jaunpur', 'jayashankar', 'jehanabad', 'jhabua', 'jhajjar', 'jhalawar', 'jhansi', 'jhargram', 'jharsuguda', 'jhunjhunu', 'jind', 'jiribam', 'jodhpur', 'jogulamba', 'jorhat', 'junagadh', 'kabirdham', 'kadapa', 'kaimur', 'kaithal', 'kakching', 'kakinada ', 'kalahandi', 'kalimpong', 'kallakurichi', 'kamareddy', 'kamjong', 'kamle', 'kamrup', 'kamrup metropolitan', 'kanchipuram', 'kandhamal', 'kangpokpi', 'kangra', 'kanker', 'kannauj', 'kannur', 'kanpur dehat', 'kanpur nagar', 'kanyakumari', 'kapurthala', 'karaikal', 'karauli', 'karbi anglong', 'kargil', 'karimganj', 'karimnagar', 'karnal', 'karur', 'kasaragod', 'kasganj', 'kathua', 'katihar', 'katni', 'kaushambi', 'kendrapara', 'kendujhar', 'khagaria', 'khairagarh', 'khammam', 'khandwa', 'khargone', 'khawzawl', 'kheda', 'kheri', 'khordha', 'khowai', 'khunti', 'kinnaur', 'kiphire', 'kishanganj', 'kishtwar', 'kodagu', 'koderma', 'kohima', 'kokrajhar', 'kolar', 'kolasib', 'kolhapur', 'kolkata', 'kollam', 'komaram bheem', 'konaseema', 'kondagaon', 'koppal', 'koraput', 'korba', 'koriya', 'kota', 'kottayam', 'kozhikode', 'kra daadi', 'krishna', 'krishnagiri', 'kulgam', 'kullu', 'kupwara', 'kurnool', 'kurukshetra', 'kurung kumey', 'kushinagar', 'kutch', 'lahaul spiti', 'lakhimpur', 'lakhisarai', 'lakshadweep', 'lalitpur', 'latehar', 'latur', 'lawngtlai', 'leh', 'lepa rada', 'lohardaga', 'lohit', 'longding', 'longleng', 'lower dibang valley', 'lower siang', 'lower subansiri', 'lucknow', 'ludhiana', 'lunglei', 'madhepura', 'madhubani', 'madurai', 'mahabubabad', 'maharajganj', 'mahasamund', 'mahbubnagar', 'mahe', 'mahendragarh', 'mahisagar', 'mahoba', 'maihar', 'mainpuri', 'mairang', 'majuli', 'malappuram', 'malda', 'malerkotla', 'malkangiri', 'mamit', 'mancherial', 'mandi', 'mandla', 'mandsaur', 'mandya', 'manendragarh', 'mansa', 'manyam', 'mathura', 'mau', 'mayiladuthurai ', 'mayurbhanj', 'medak', 'medchal', 'meerut', 'mehsana', 'mewat', 'mirzapur', 'moga', 'mohali', 'mohla manpur', 'mokokchung', 'mon', 'moradabad', 'morbi', 'morena', 'morigaon', 'muktsar', 'mulugu', 'mumbai city', 'mumbai suburban', 'mungeli', 'munger', 'murshidabad', 'muzaffarnagar', 'muzaffarpur', 'mysore', 'n t rama rao', 'nabarangpur', 'nadia', 'nagaon', 'nagapattinam', 'nagarkurnool', 'nagaur', 'nagda', 'nagpur', 'nainital', 'nalanda', 'nalbari', 'nalgonda', 'namakkal', 'namsai', 'nanded', 'nandurbar', 'nandyal', 'narayanpet', 'narayanpur', 'narmada', 'narsinghpur', 'nashik', 'navsari', 'nawada', 'nayagarh', 'neemuch', 'nellore', 'new delhi', 'nicobar', 'nilgiris', 'nirmal', 'niuland', 'niwari', 'nizamabad', 'noklak', 'noney', 'north 24 parganas', 'north delhi', 'north east delhi', 'north garo hills', 'north goa', 'north middle andaman', 'north sikkim', 'north tripura', 'north west delhi', 'nuapada', 'osmanabad', 'pakke kessang', 'pakur', 'pakyong', 'palakkad', 'palamu', 'palghar', 'pali', 'palnadu', 'palwal', 'panchkula', 'panchmahal', 'panipat', 'panna', 'papum pare', 'parbhani', 'paschim bardhaman', 'paschim medinipur', 'patan', 'pathanamthitta', 'pathankot', 'patiala', 'patna', 'pauri', 'peddapalli', 'perambalur', 'peren', 'phek', 'pherzawl', 'pilibhit', 'pithoragarh', 'poonch', 'porbandar', 'prakasam', 'pratapgarh', 'prayagraj', 'puducherry', 'pudukkottai', 'pulwama', 'pune', 'purba bardhaman', 'purba medinipur', 'puri', 'purnia', 'purulia', 'raebareli', 'raichur', 'raigad', 'raigarh', 'raipur', 'raisen', 'rajanna sircilla', 'rajgarh', 'rajkot', 'rajnandgaon', 'rajouri', 'rajsamand', 'ramanagara', 'ramanathapuram', 'ramban', 'ramgarh', 'rampur', 'ranchi', 'ranga reddy', 'ranipet', 'ratlam', 'ratnagiri', 'rayagada', 'reasi', 'rewa', 'rewari', 'ri bhoi', 'rohtak', 'rohtas', 'rudraprayag', 'rupnagar', 'sabarkantha', 'sagar', 'saharanpur', 'saharsa', 'sahebganj', 'saiha', 'saitual', 'sakti', 'salem', 'samastipur', 'samba', 'sambalpur', 'sambhal', 'sangareddy', 'sangli', 'sangrur', 'sant kabir nagar', 'saran', 'sarangarh bilaigarh', 'satara', 'satna', 'sawai madhopur', 'sehore', 'senapati', 'seoni', 'sepahijala', 'seraikela kharsawan', 'serchhip', 'shahdara', 'shahdol', 'shaheed bhagat singh nagar', 'shahjahanpur', 'shajapur', 'shamator', 'shamli', 'sheikhpura', 'sheohar', 'sheopur', 'shi yomi', 'shimla', 'shimoga', 'shivpuri', 'shopian', 'shravasti', 'siddharthnagar', 'siddipet', 'sidhi', 'sikar', 'simdega', 'sindhudurg', 'singrauli', 'sirmaur', 'sirohi', 'sirsa', 'sitamarhi', 'sitapur', 'sivaganga', 'sivasagar', 'siwan', 'solan', 'solapur', 'sonbhadra', 'sonipat', 'sonitpur', 'soreng', 'south 24 parganas', 'south andaman', 'south delhi', 'south east delhi', 'south garo hills', 'south goa', 'south salmara-mankachar', 'south sikkim', 'south tripura', 'south west delhi', 'south west garo hills', 'south west khasi hills', 'sri balaji', 'sri ganganagar', 'sri satya sai', 'srikakulam', 'srinagar', 'subarnapur', 'sukma', 'sultanpur', 'sundergarh', 'supaul', 'surajpur', 'surat', 'surendranagar', 'surguja', 'suryapet', 'tamenglong', 'tapi', 'tarn taran', 'tawang', 'tehri', 'tengnoupal', 'tenkasi', 'thane', 'thanjavur', 'theni', 'thiruvananthapuram', 'thoothukudi', 'thoubal', 'thrissur', 'tikamgarh', 'tinsukia', 'tirap', 'tiruchirappalli', 'tirunelveli', 'tirupattur', 'tiruppur', 'tiruvallur', 'tiruvannamalai', 'tiruvarur', 'tonk', 'tseminyu', 'tuensang', 'tumkur', 'udaipur', 'udalguri', 'udham singh nagar', 'udhampur', 'udupi', 'ujjain', 'ukhrul', 'umaria', 'una', 'unakoti', 'unnao', 'upper siang', 'upper subansiri', 'uttar dinajpur', 'uttara kannada', 'uttarkashi', 'vadodara', 'vaishali', 'valsad', 'varanasi', 'vellore', 'vidisha', 'vijayanagara', 'vijayapura ', 'vikarabad', 'viluppuram', 'virudhunagar', 'visakhapatnam', 'vizianagaram', 'wanaparthy', 'warangal', 'wardha', 'washim', 'wayanad', 'west champaran', 'west delhi', 'west garo hills', 'west godavari', 'west jaintia hills', 'west kameng', 'west karbi anglong', 'west khasi hills', 'west siang', 'west sikkim', 'west singhbhum', 'west tripura', 'wokha', 'yadadri bhuvanagiri', 'yadgir', 'yamunanagar', 'yanam', 'yavatmal', 'zunheboto'}
        
        #Converting all the string values present in the df into lowercase
        data = data.applymap(lambda x: x.lower() if isinstance(x, str) else x)

        #converting the columns names into lowercase
        data.columns = data.columns.str.lower()
        
        #storing data columns
        data_cols = set()
        for items in data.columns:
            data_cols.add(items)
        
        #Making the state value None if wrong input is present in the name of a state
        if 'state' in data_cols:
            for i, item in enumerate(data['state']):
                if item not in states_dict:
                    data.at[i, 'state'] = None
        
        if 'district' in data_cols:
            #Making the state value None if wrong input is present in the name of a state
            for i, item in enumerate(data['district']):
                if item not in all_districts:
                    data.at[i, 'district'] = None

        if 'pincode' in data_cols:
            #Making the state value None if wrong input is present in the name of a state
            for i, item in enumerate(data['district']):
                if len(str(item)) != 6:
                    data.at[i, 'pincode'] = None

        return data

    #Function to check whether any string is present in a numeric type column
    def clean_column(data_set):
        data = data_set.copy()
        for column_name in data.columns:
            # Calculate the percentage of numeric values in the column
            numeric_percentage = (data[column_name].apply(lambda x: isinstance(x, (int, float))).sum() / len(data)) * 100
            char_percentage = (data[column_name].apply(lambda x: isinstance(x, str)).sum() / len(data)) * 100
            # Check if the percentage of numeric values is above 90
            if numeric_percentage > 93:
                # Set non-numeric values to None
                data.loc[~data[column_name].apply(lambda x: isinstance(x, (int, float))), column_name] = None

            if char_percentage > 93 :
                data.loc[~data[column_name].apply(lambda x: isinstance(x, str)), column_name] = None
        return data

    def shape_address(data_set):
        data = data_set.copy()
        """
        Shape the "Address" column values in a pandas DataFrame into a format that can be used to find the latitude and longitude values.
        """
        # define a function to clean and shape the address string
        def clean_address(address):
            # replace any multiple spaces with a single space
            address = re.sub('\s+', ' ', str(address))
            # remove any leading or trailing spaces
            address = address.strip()
            # remove any commas
            address = address.replace(',', '').replace('/', '').replace('%', '')
            # replace any double quotes with single quotes
            address = address.replace('"', "'")
            # return the cleaned address string
            return address
        
        # iterate over the rows of the DataFrame
        for index, row in data.iterrows():
            # get the address from the "Address" column
            address = row["address"]
            # if the address is missing, skip this row
            if pd.isna(address):
                continue
            # clean and shape the address string using the clean_address function
            address = clean_address(address)
            # update the DataFrame with the new value
            data.at[index, "address"] = address
        
        # return the updated DataFrame
        return data

    # Calling the function for every column that user have selected to make the minority None based on variable type present

    data1 = clean_column(data1)

    if 'address' in data1.columns:
        data1 = shape_address(data1)
    
    #cleaning state, district, pincode  values in the df
    if 'state' in data1.columns or 'district' in data1.columns or 'pincode' in data1.columns:
        data1 = state_district(data1)

#-------First focusing to fill the null values of state, pincode, district, latitude and longitude values of a df if these columns are present 
    
    #Function to impute latitude-longitude values using pincode or address
    def impute_lat_long(data_set):
        data = data_set.copy()
        """
        Impute missing latitude and longitude values using the "pincode" and "address" columns in a pandas DataFrame.
        """
        # define a function to get the latitude and longitude from an address using the OpenStreetMap API
        def get_lat_long(address):
            url = f"https://nominatim.openstreetmap.org/search?q={address}&format=json"
            response = requests.get(url).json()
            if len(response) > 0:
                return response[0]["lat"], response[0]["lon"]
            else:
                return None, None
        
        # iterate over the rows of the DataFrame
        for index, row in data.iterrows():
            # check if the latitude or longitude value is missing
            if pd.isna(row["latitude"]) or pd.isna(row["longitude"]):
                # get the address from the "Address" column
                if 'address' in data.columns:
                    address = row["address"]
                else:
                    address = None
                # if the address is missing, try to use the pincode
                if pd.isna(address):
                    pincode = row["pincode"]
                    if pd.notna(pincode):
                        address = str(pincode)
                # if the address is not missing, add the pincode to it
                else:
                    pincode = row["pincode"]
                    if pd.notna(pincode):
                        address += " " + str(pincode)
                # if the address is still missing, skip this row
                if pd.isna(address):
                    continue
                # get the latitude and longitude from the address using the get_lat_long function
                lat, long = get_lat_long(address)
                # update the DataFrame with the new values
                data.at[index, "latitude"] = lat
                data.at[index, "longitude"] = long
        
        # return the updated DataFrame
        return data

    #---function to impute missing pincode, state, district values 
    def check_and_impute_pin_dist_state(data_set):
        data = data_set.copy()
        """
        A function that imputes pincode, district and state column values using the pincode or latitude-longitude values.

        Parameters:
        df (pandas.DataFrame): The pandas DataFrame containing columns: state, district, pincode, address_line_one, latitude, longitude.

        Returns:
        df (pandas.DataFrame): The modified pandas DataFrame with filled in state and district values.
        """
        data.columns = map(str.lower, data.columns) #converting column names into lowercase
        # Initialize a geolocator object
        geolocator = Nominatim(user_agent="geoapiExercises")

            # Loop through each row in the DataFrame
        for index, row in data.iterrows():
            # Check if the pincode is missing or None
            if pd.isna(row['pincode']) or row['pincode'] is None:
                if not pd.isna(row['latitude']) and row['longitude'] is not None:
                    # Combine the latitude and longitude values into a single string
                    location = str(row['latitude']) + "," + str(row['longitude'])

                    # Try to obtain the pincode value using the geolocator object
                    try:
                        # Use reverse geocoding to get the address details
                        location_details = geolocator.reverse(location, exactly_one=True)
                        # Extract the pincode value from the address details
                        Pincode = location_details.raw['address']['postcode']
                        if location_details.raw['address']['state']:
                            state = location_details.raw['address']['state']
                    except (AttributeError, GeocoderTimedOut, KeyError):
                        # If an error occurs, set the pincode to None
                        Pincode = None


                    # Fill in the missing pincode value with the obtained pincode
                    data.loc[index, 'pincode'] = Pincode
#                    data.loc[index, 'state'] = state

            else:
                # Check if the pincode is valid using a regular expression
                if not pd.isna(row['pincode']) and row['pincode'] is not None:
                    if not re.match(r'^\d{6}$', str(row['pincode'])):
                        if not pd.isna(row['latitude']) and row['longitude'] is not None:
                            # Combine the latitude and longitude values into a single string
                            location = str(row['latitude']) + "," + str(row['longitude'])
                        else:
                            location = str(row['pincode'])
                        # Try to obtain the pincode value using the geolocator object
                        try:
                            # Use reverse geocoding to get the address details
                            location_details = geolocator.reverse(location, exactly_one=True)
                            # Extract the pincode value from the address details
                            Pincode = location_details.raw['address']['postcode']
                            state = location_details.raw['address']['state']

                        except (AttributeError, GeocoderTimedOut, KeyError):
                            # If an error occurs, set the pincode to None
                            Pincode = None

                        # Fill in the incorrect pincode value with the obtained pincode
                        data.loc[index, 'pincode'] = Pincode
                        data.loc[index, 'state'] = state

        data['pincode'] = data['pincode'].fillna(0)
        data['pincode'] = data['pincode'].astype(int)
        # Loop through each row in the DataFrame
        for index, row in data.iterrows():
            # Check if the pincode is missing or None
            if pd.isna(row['district']) or row['district'] is None :
                if not pd.isna(row['pincode']) and row['pincode'] is not None:
                    # Try to obtain the address details using the geolocator object
                    try:
                        ENDPOINT = 'https://api.postalpincode.in/pincode/'
                        response = requests.get(ENDPOINT + str(row['pincode']))
                    except (AttributeError, GeocoderTimedOut, KeyError):
                        # If an error occurs, set the address details to None
                        response = None

                    # Extract the state and district values from the address details
                    if response is not None:
                        try:
                            info = json.loads(response.text)[0]['PostOffice']
                            if info is not None:
                                data.loc[index, 'district'] = info[0]['District']
                                data.loc[index, 'state'] = info[0]['State']
                            else:
                                try:
                                    location_details = geolocator.geocode(row['pincode'], exactly_one=True)
                                except (AttributeError, GeocoderTimedOut, KeyError):
                                    # If an error occurs, set the address details to None
                                    location_details = None

                                # Extract the state and district values from the address details
                                if location_details is not None:
                                    try:
                                        data.loc[index, 'district'] = location_details.raw['address']['county']
                                        data.loc[index, 'dtate'] = location_details.raw['address']['state']
                                    except KeyError:
                                        pass 
                        except KeyError:
                            pass
        print("I have sucessfully imputed pincode-state-districts values up to my potential.")
        data['pincode'] = data['pincode'].apply(lambda x: None if x == 0 else x)
        return data


    #function to fill null values
    def fillGenColumns_null(data_set):
        data = data_set.copy()
        """
        Fill the missing values (None) in a DataFrame column based on the data type of the column.
        For numeric columns, fill with the average value.
        For string/character columns, fill with the most frequent value.
        """
        clean_column(data1)
        dict_avoid_cols = {'state':'1', 'pincode':'2', 'latitude':'3', 'longitude':'4', 'district':'5', 'address':'6'}
        # Iterate over each column in the DataFrame
        for column in data.columns:
            # Check if the column contains None values
            
            if column not in dict_avoid_cols: 
                if data[column].isnull().any():
                    # Check the data type of the column
                    if pd.api.types.is_numeric_dtype(data[column]):
                        # Fill missing values with the average value of the column
                        average = data[column].mean()
                        if math.isnan(average):
                            pass
                        else:
                            x = average - int(average)
                            if x > 0.5:
                                data[column] = data[column].fillna(int(average+1))
                            else:
                                data[column] = data[column].fillna(int(average))
                    else:
                        # Fill missing values with the most frequent value of the column
                        most_frequent = data[column].mode().iloc[0]
                        data[column] = data[column].fillna(most_frequent)
        # Return the updated DataFrame
        return data
    
    def fill_address(data_set):
        df = data_set.copy()
        for index, row in df.iterrows():
            if row['address'] == None: 
                pincode = row['pincode']
                district = row['district']
                state = row['state']
                if pincode == None:
                    pincode = " "
            
                if district == None:
                    district = " "
                
                if state == None:
                    state = " "

                # Check if pincode, district, and state are not None
                if pincode is not None or district is not None or state is not None:
                    # Form the address using pincode, district, and state
                    address = str(pincode) + " " + (district) +  " " + (state)
                    df.at[index, 'address'] = address

        return df
    

    def shape_address(data_set):
        data = data_set.copy()
        """
        Shape the "Address" column values in a pandas DataFrame into a format that can be used to find the latitude and longitude values.
        """
        # define a function to clean and shape the address string
        def clean_address(address):
            # replace any multiple spaces with a single space
            address = re.sub('\s+', ' ', str(address))
            # remove any leading or trailing spaces
            address = address.strip()
            # remove any commas
            address = address.replace(',', '').replace('/', '').replace('%', '')
            # replace any double quotes with single quotes
            address = address.replace('"', "'")
            # return the cleaned address string
            return address
        
        # iterate over the rows of the DataFrame
        for index, row in data.iterrows():
            # get the address from the "Address" column
            address = row["address"]
            # if the address is missing, skip this row
            if pd.isna(address):
                continue
            # clean and shape the address string using the clean_address function
            address = clean_address(address)
            # update the DataFrame with the new value
            data.at[index, "address"] = address
        
        # return the updated DataFrame
        return data
    
    #extracting pincode from address column
    def fill_pincode(data_set):
        data = data_set.copy()
        """
        Fill missing values in the 'pincode' column by extracting pincode from the corresponding 'address' cell.
        """
        for index, row in data.iterrows():
            address = str(row['address'])
            pincode = row['pincode']
            if address == None:
                address = 'not_found'
            
            if pd.isnull(pincode):  # Check if pincode is null
                
                extracted_pincode = re.findall(r"\b\d{6}\b", address)  # Extract pincode from address
                
                if extracted_pincode:  # Check if pincode is found in address
                    data.at[index, 'pincode'] = extracted_pincode[0]  # Fill pincode cell with extracted pincode
        
        return data

    #filling the null address the address 
    if 'address' in data1.columns:
        data1 = fill_address(data1)

    #shaping the address 
    if 'address' in data1.columns:
        print('shaping address')
        data1 = shape_address(data1)

    #extracting the null values from the address
    if ("pincode" and 'address') in data1.columns:
        print('filling pincode')
        data1 = fill_pincode(data1)

    #filling the latitude and longitude
    if ('latitude' and 'longitude') in finalCols:
        if 'pincode' in data1.columns or 'address' in data1.columns :
            print('filling lat-long')
            data1 = impute_lat_long(data1)
    
#    df = fillGenColumns_null(df)
    ans = data1['state'][2:4].to_list()
    print(ans)

    #imputing pincode, state, district
    if ('latitude' and 'longitude') in data1.columns:
        if ('pincode' in data1.columns) or ('state' in data1.columns) or ('district' in data1.columns):
            data1 = check_and_impute_pin_dist_state(data1)


    df = data1.copy()
    # Convert DataFrame to JSON and return as response

    df = fillGenColumns_null(df)
    ans = df['pincode'][2:4].to_list()
    print(ans)
    df_1 = df.copy()
    print("I'm ready to show the results")
    # Convert DataFrame to JSON and return as response
    data = df_1.to_dict(orient='records')

    with open("download.html", "r") as file:
        html_content = file.read()

    # Encode the data as JSON and insert it into the HTML content
    json_data = json.dumps(data)
    html_content = html_content.replace("{{ data }}", json_data)

    return response.html(html_content)
#    data = json_data1.copy()
#    return response.json(data)

#----API to remove the none values----------------

@app.route("/remove_null", methods=["GET"])
async def remove_null_values(request:Request):
     
    print("remove null is getting executed")
    global df


    data1 = df.copy()
    def remove_null_values(df):
        """
        Remove rows containing None values from a DataFrame.
        """
        # Drop rows with None values
        df = df.dropna()
        
        # Reset the index of the DataFrame
        df = df.reset_index(drop=True)
        
        # Return the updated DataFrame
        return df
    data1 = remove_null_values(data1)
    df = data1.copy()
    # Convert DataFrame to JSON and return as response
    json_data2 = df.to_json(orient="records")
    return response.json(json_data2)


#-----API to visualize the final processed data --------

@app.route("/data", methods=["GET"])
async def get_data(request):
    global df
    global df_1
    global finalCols

    if df_1 is None :
        df1 = df_original.copy()
    else:
        df1 = df_1.copy()

    selected_columns1 = []
    for items in finalCols.keys():
        selected_columns1 += [items]

    df1 = df1.filter(items = selected_columns1)
    df_1 = df1.copy()
    df_1 = df_1.fillna('none')
    data = df_1.to_dict(orient='records')
    return response.json(data)

@app.route("/download", methods=["GET"])
async def serve_html(request):
    return await response.file("download.html")

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000)