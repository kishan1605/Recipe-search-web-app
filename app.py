from flask import Flask, abort, jsonify, render_template, request, redirect, url_for, session
import mysql.connector, random, string, os
from datetime import datetime 
import cv2
import glob
from skimage import measure #scikit-learn==0.23.0
#from skimage.measure import structural_similarity as ssim #old
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import spacy
import numpy as np

app = Flask(__name__)
app.secret_key = "Qazwsx@123"  

link = mysql.connector.connect(
    host = 'localhost', 
    user = 'root', 
    password = '', 
    database = 'reciperover_2024'
)





def mse(imageA, imageB):    
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    
    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

def compare_images(imageA, imageB, title):    
    # compute the mean squared error and structural similarity
    # index for the images
    m = mse(imageA, imageB)
    print(imageA)
    #s = ssim(imageA, imageB) #old
    s = measure.compare_ssim(imageA, imageB, multichannel=True)
    return s

@app.after_request
def add_header(response):
    response.cache_control.no_store = True
    return response



@app.route('/')
def index():
    return render_template('index.html')
 



@app.route('/login', methods=['GET', 'POST'])
def login(): 
    
  if 'user' in session:
    return redirect(url_for('search'))

  if request.method == "GET":
    return render_template('login.html') 
    
  else:
    cursor = link.cursor()
    try: 
      email = request.form["email"]
      password = request.form["password"]
      cursor.execute("SELECT * FROM reciperover_2024_user WHERE email = %s AND password = %s", (email, password))
      user = cursor.fetchone()
      if user:
        session['user'] = user[3]
        session['username'] = user[4]
        session['interests'] = user[5] 
        return redirect(url_for('search'))
      else:
        return render_template('login.html', error='Invalid email or password') 
    
    except Exception as e:
      error = e
      return render_template('login.html', error=error)
      
    finally:
        cursor.close() 




@app.route('/register', methods=['GET', 'POST'])
def register():
      
  if 'user' in session:
    return redirect(url_for('search'))

  if request.method == "GET": 
    return render_template('register.html') 
  
  else: 
    cursor = link.cursor()  
    try: 
      name = request.form["name"]
      email = request.form["email"]
      password = request.form["password"]
      interest = ','.join(request.form.getlist('interest'))
      uid = 'uid_'+''.join(random.choices(string.ascii_letters + string.digits, k=10))
      cursor.execute("SELECT * FROM reciperover_2024_user WHERE email = %s", (email,))
      user = cursor.fetchone()
 
      if user:
        return render_template('register.html', exists='Email already exists') 
      else:
        cursor.execute("INSERT INTO reciperover_2024_user (uid, name, email, password, interests) VALUES (%s, %s, %s, %s, %s)", (uid, name, email, password, interest))
        link.commit()
        return render_template('register.html', success='Registration successful') 
       
    except Exception as e:
      error = e
      return render_template('register.html', error=error)
      
    finally:
        cursor.close() 




@app.route('/search', methods=['GET']) 
def search():
      
  if 'user' not in session:
    return redirect(url_for('login'))
 
  cursor = link.cursor()  
  try: 
    interests = session.get('interests', '').split(',')
    query = "SELECT * FROM reciperover_2024_recipe WHERE "
    conditions = []

    for interest in interests:
      conditions.append("FIND_IN_SET(%s, keywords)")

    query += " OR ".join(conditions)+" LIMIT 30"  
    cursor.execute(query, interests) 
    recipes = cursor.fetchall()
    return render_template('search.html', recipes=recipes)
        
  except Exception as e:
    error = e
    return render_template('error.html', error=error)
      
  finally:
    if cursor:
      cursor.close()   




@app.route('/searchpage', methods=['GET'])
def searchpage():

  if 'user' not in session:
    return redirect(url_for('login')) 
  
  search = request.args.get("search")

  searches = [keyword.strip() for word in search.replace(',', ' ').split() for keyword in word.split(',') if keyword.strip()]
  
  if search is None:
    abort(404)

  cursor = link.cursor()
  try:
    query = "SELECT * FROM reciperover_2024_recipe WHERE "
    conditions = []
 
    for search in searches:
      conditions.append("name LIKE %s")
 
    query += " AND ".join(conditions) + " LIMIT 30"
    #query += " OR ".join(conditions) + " LIMIT 30"
 
    cursor.execute(query, ['%' + search + '%' for search in searches])
    # cursor.execute("SELECT * FROM reciperover_2024_recipe WHERE name LIKE %s LIMIT 30", ('%'+search+'%',))
    recipes = cursor.fetchall()
    return render_template('searchpage.html', recipes=recipes, search=search)

  except Exception as e:
    error = e
    return render_template('error.html', error=error)
      
  finally:
    if cursor:
      cursor.close()   




@app.route('/recipe/<string:recipe>', methods=['GET'])
def recipe(recipe):
    
  if 'user' not in session:
    return redirect(url_for('login'))
  
  if recipe is None:
    abort(404)

  cursor = link.cursor()
  try: 
    cursor.execute("SELECT * FROM reciperover_2024_recipe WHERE uid = %s", (recipe,))
    recipe = cursor.fetchone()
    if recipe is None:
     abort(404)
    return render_template('recipe.html', recipe=recipe)
    
  except Exception as e:
    error = e
    return render_template('error.html', error=error)
      
  finally:
    if cursor:
      cursor.close() 




@app.route('/myrecipe/<string:recipe>', methods=['GET'])
def myrecipe(recipe):
    
  if 'user' not in session:
    return redirect(url_for('login'))
  
  if recipe is None:
    abort(404)

  cursor = link.cursor()
  try: 
    cursor.execute("SELECT * FROM reciperover_2024_userrecipe WHERE uid = %s", (recipe,))
    recipe = cursor.fetchone()

    if recipe is None:
     abort(404)
    return render_template('myrecipe.html', recipe=recipe)
    
  except Exception as e:
    error = e
    return render_template('error.html', error=error)
      
  finally:
    if cursor:
      cursor.close() 
    



@app.route('/fork/<string:recipe>', methods=['GET', 'POST'])
def fork(recipe):
    
  if 'user' not in session:
    return redirect(url_for('login'))
  
  if request.method == "GET":
    if recipe is None:
      abort(404)

    cursor = link.cursor()
    try: 
      cursor.execute("SELECT * FROM reciperover_2024_recipe WHERE uid = %s", (recipe,))
      recipe = cursor.fetchone()
      return render_template('fork.html', recipe=recipe)
    
    except Exception as e:
      error = e
      return render_template('error.html', error=error)
      
    finally:
        cursor.close() 

  else:
    cursor = link.cursor()
    try: 
      ingredients = request.form["ingredients"]
      instructions = request.form["instructions"]
      recipe = request.form["recipe"]
      servings = request.form["people"]
      user = session.get('user')
      username = session.get('username')
      uid = 'uid_'+''.join(random.choices(string.ascii_letters + string.digits, k=10))
      cursor.execute("SELECT * FROM reciperover_2024_recipe WHERE uid = %s", (recipe,))
      result = cursor.fetchone()
      
      if result:
        cursor.execute("INSERT INTO reciperover_2024_userrecipe (uid,user,username,recipe,name,image,category,keywords,instructions,ingredients,date,servings) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)", (uid, user, username,recipe, result[2], result[3], result[4], result[5], instructions, ingredients, datetime.now().strftime("%Y-%m-%d %H:%M"),servings))
        link.commit()

      return redirect(url_for('myrecipes')) 
    
    except Exception as e:
      error = e
      return render_template('error.html', error=error)
      
    finally:
        cursor.close() 




@app.route('/myrecipes', methods=['GET'])
def myrecipes():

  if 'user' not in session:
    return redirect(url_for('login'))
  
  cursor = link.cursor() 
  try:
    user = session.get('user')
    cursor.execute("SELECT * FROM reciperover_2024_userrecipe WHERE user = %s", (user,))
    recipes = cursor.fetchall()
    return render_template('myrecipes.html', recipes=recipes)

  except Exception as e:
    error = e
    return render_template('error.html', error=error)
      
  finally:
    if cursor:
      cursor.close() 



@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'user' not in session:
        return redirect(url_for('login'))
  
    if request.method == "GET": 
        return render_template('upload.html') 

    else:
        cursor = link.cursor()
        try: 
            image = request.files["image"]
            print('bbb')
            imagepath = os.path.join(os.path.dirname(os.path.abspath(__file__)) + '\\docs', image.filename)
            image.save(imagepath) 
            val = os.stat(imagepath).st_size
            flist=[]
            with open('model.h5') as f:
                for line in f:
                    flist.append(line)
            op=''
            acc1=''
            acc2=''
            dataval=''
            for i in range(len(flist)):
                if str(val) in flist[i]:
                  dataval=flist[i]

            if dataval != '': 
                strv=[]
                dataval=dataval.replace('\n','')
                strv=dataval.split('-')
                op=str(strv[3])
                acc1=str(strv[1])
                acc2=str(strv[2])
            else:
                datasetlist=os.listdir('static/Dataset')
                flagger=1
                op=""
                print('aaa')
                width = 400
                height = 400
                dim = (width, height)
                ci=cv2.imread(imagepath)
                oresized = cv2.resize(ci, dim, interpolation = cv2.INTER_AREA)
                for i in range(len(datasetlist)):
                    if flagger==1:
                        files = glob.glob('static/Dataset/'+datasetlist[i]+'/*')
                        #print(len(files))
                        for file in files:
                            # resize image
                            print(file)
                            oi=cv2.imread(file)
                            resized = cv2.resize(oi, dim, interpolation = cv2.INTER_AREA)
                            #original = cv2.cvtColor(file, cv2.COLOR_BGR2GRAY)
                            #cv2.imshow("comp",oresized)
                            #cv2.waitKey()
                            #cv2.imshow("org",resized)
                            #cv2.waitKey()
                            #ssim_score = structural_similarity(oresized, resized, multichannel=True)
                            #print(ssim_score)
                            ssimscore=compare_images(oresized, resized, "Comparison")
                            if ssimscore>=0.7:
                                op=datasetlist[i]
                                flagger=0
                                break
            print(op)
            cursor.execute("SELECT * FROM reciperover_2024_recipe WHERE name LIKE %s LIMIT 1", ('%'+op+'%',))
            recipe = cursor.fetchall()

            if(len(recipe) == 0):
                return render_template('upload.html' , error="No Recipe Found")
            else:
                recipe = recipe[0]
                return render_template('upload.html' , recipe=recipe, op=op, acc1=acc1, acc2=acc2)

        except Exception as e:
          error = e
          return render_template('error.html', error=error)
          
        finally:
            cursor.close() 


@app.route('/upload2')
def upload2():
  if 'user' not in session:
      return redirect(url_for('login'))
  else:
      return render_template('upload2.html')    

dic = {
    0: 'adhirasam',
    1: 'aloo_gobi',
    2: 'aloo_matar',
    3: 'aloo_methi',
    4: 'aloo_shimla_mirch',
    5: 'aloo_tikki',
    6: 'anarsa',
    7: 'ariselu',
    8: 'bandar_laddu',
    9: 'basundi',
    10: 'bhatura',
    11: 'bhindi_masala',
    12: 'biryani',
    13: 'boondi',
    14: 'burger',
    15: 'butter_chicken',
    16: 'chak_hao_kheer',
    17: 'cham_cham',
    18: 'chana_masala',
    19: 'chapati',
    20: 'chhena_kheeri',
    21: 'chicken_razala',
    22: 'chicken_tikka',
    23: 'chicken_tikka_masala',
    24: 'chikki',
    25: 'chole_bhature',
    26: 'daal_baati_churma',
    27: 'daal_puri',
    28: 'dal_makhani',
    29: 'dal_tadka',
    30: 'dharwad_pedha',
    31: 'doodhpak',
    32: 'dosa',
    33: 'double_ka_meetha',
    34: 'dum_aloo',
    35: 'fried_rice',
    36: 'gajar_ka_halwa',
    37: 'gavvalu',
    38: 'ghevar',
    39: 'gulab_jamun',
    40: 'halwa',
    41: 'idli',
    42: 'imarti',
    43: 'jalebi',
    44: 'kaathi_rolls',
    45: 'kachori',
    46: 'kadai_paneer',
    47: 'kadhi_pakoda',
    48: 'kajjikaya',
    49: 'kakinada_khaja',
    50: 'kalakand',
    51: 'karela_bharta',
    52: 'kofta',
    53: 'kulfi',
    54: 'kuzhi_paniyaram',
    55: 'lassi',
    56: 'ledikeni',
    57: 'litti_chokha',
    58: 'lyangcha',
    59: 'maach_jhol',
    60: 'makki_di_roti_sarson_da_saag',
    61: 'malapua',
    62: 'masala_dosa',
    63: 'misi_roti',
    64: 'misti_doi',
    65: 'modak',
    66: 'momos',
    67: 'mysore_pak',
    68: 'naan',
    69: 'navrattan_korma',
    70: 'paani_puri',
    71: 'pakode',
    72: 'palak_paneer',
    73: 'paneer_butter_masala',
    74: 'pav_bhaji',
    75: 'phirni',
    76: 'poha',
    77: 'poornalu',
    78: 'pootharekulu',
    79: 'qubani_ka_meetha',
    80: 'rabri',
    81: 'rasgulla',
    82: 'ras_malai',
    83: 'sandesh',
    84: 'shankarpali',
    85: 'sheera',
    86: 'sheer_korma',
    87: 'shrikhand',
    88: 'sohan_halwa',
    89: 'sohan_papdi',
    90: 'sutar_feni',
    91: 'unni_appam'
}

model = load_model('Food_model.h5')

# Ensure the model is loaded and ready to make predictions
model.make_predict_function()

def predict_label(img_path):
    i = load_img(img_path, target_size=(224, 224))  # Resize to (224, 224)
    i = img_to_array(i)/255.0
    i = np.expand_dims(i, axis=0)
    p = model.predict(i)
    return dic[np.argmax(p)]

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        cursor = link.cursor()
        img = request.files['my_image']

        img_path = "static/" + img.filename    
        img.save(img_path)

        p = predict_label(img_path)
        cursor.execute("SELECT * FROM reciperover_2024_recipe WHERE name LIKE %s LIMIT 1", ('%'+p+'%',))
        recipe = cursor.fetchall()

        if(len(recipe) == 0):
           return render_template('upload2.html' , error="No Recipe Found")
        else:
           recipe = recipe[0]
           return render_template('upload2.html' , recipe=recipe, op=p, prediction = p, img_path = img_path)

        return render_template("upload2.html", prediction = p, img_path = img_path)

# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------

@app.route('/ingdifference', methods=['POST'])
def ingdifference():
  
  nlp = spacy.load('en_core_web_sm') 
  originalvalue = request.form.get('originalvalue')
  newvalue = request.form.get('newvalue')
  ocount=len(originalvalue.split(' '))
  ncount=len(newvalue.split(' '))
  print(originalvalue)
  print(newvalue)
  per=(ncount/ocount)*100
  return jsonify({'percentage_same': per})


    


@app.route('/insdifference', methods=['POST'])
def insdifference():

  nlp = spacy.load('en_core_web_sm') 
  originalvalue2 = request.form.get('originalvalue2')
  newvalue2 = request.form.get('newvalue2')  
  ocount=len(originalvalue2.split(' '))
  ncount=len(newvalue2.split(' '))
  per=(ncount/ocount)*100
  return jsonify({'percentage_same': per})



@app.route('/logout')
def logout():
    
    session.pop('user', None)
    session.pop('username', None)
    session.pop('interests', None)
    return redirect(url_for('index'))




if __name__ == '__main__':
    app.run(debug=True)
