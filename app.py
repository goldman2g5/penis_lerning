from flask import Flask, render_template, request, redirect
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
    message = 'Для входа введите логин и пароль'
    login = request.form.get('username')  # запрос к данным формы
    password = request.form.get('password')
    if login != None or password != None:
        cur = dbh.cursor()
        cur.execute(f'SELECT login, password FROM users WHERE login = "{login}" AND password = "{password}";')
        a = cur.fetchone()
        if a:
            if a['login'] == login and a['password'] == password:
                return redirect("/neyronka")
        else:
            message = "Wrong username or password"
    return render_template('login.html', message=message)

@app.route('/reg/', methods=['post', 'get'])
def registration():
    message = 'Для регистрации заполните форму ниже'

    login = request.form.get('login')  # запрос к данным формы
    password = request.form.get('password')
    username = request.form.get('username')
    print(f'l - {login},p - {password},u - {username}')
    print(f'l - {type(login)},p - {type(password)},u - {type(username)}')
    if login and password and username and len(login) >= 4 and len(password) >= 4 and len(username) >= 4:
        cur = dbh.cursor()
        cur.execute(f'SELECT login, password FROM users WHERE login = "{login}" OR password = "{password}" OR name = "{username}";')
        b = cur.fetchone()
        print(f'b = {b}')
        if b == None:
            cur.execute(f'INSERT INTO users (id, login, password, name) VALUES (NULL, "{login}", "{password}", "{username}");')
            dbh.commit()
            message = "Успешная регистрация"
            return redirect("/login")
        else:
            message = "Логин, пароль или имя уже используются, попробуйте другие"

    return render_template('reg.html', message = message)


@app.route('/neyronka/', methods=['post', 'get'])
def huy():
    return render_template('neyronka.html')


@app.route('/user/<string:name>/<int:id>')
def user(name, id):
    return "User page: " + name + " - " + str(id)


if __name__ == "__main__":
    app.run(debug=True)

