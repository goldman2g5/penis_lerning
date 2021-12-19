from flask import Flask, render_template, request
import pymysql
from pymysql.cursors import DictCursor

app = Flask(__name__)

dbh = pymysql.connect(
        host='185.12.94.106',
        user='2p1s23',
        password='941-961-748',
        db='2p1s23',
        charset='utf8mb4',
        cursorclass = DictCursor
    )

@app.route('/login/', methods=['post', 'get'])
def login():
    message = ''
    username = request.form.get('username')  # запрос к данным формы
    password = request.form.get('password')
    cur = dbh.cursor()
    a = cur.execute(f'SELECT login, password FROM users WHERE login = "{username}" AND password = "{password}";')

    if a == 1:
        message = "Correct username and password"
    elif a == 0:
        message = "Wrong username or password"
    return render_template('login.html', message=message)



@app.route('/user/<string:name>/<int:id>')
def user(name, id):
    return "User page: " + name + " - " + str(id)


if __name__ == "__main__":
    app.run(debug=True)

