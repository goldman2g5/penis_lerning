"""
Анекдоты:

    Анекдот № 1:
    Прыгает девочка на скакалке, рядом на лавочку подсаживается дедок. Девочка:
    — Дедуль, скажи раз…
    Дед:
    — Раз…
    — Ты, дед, пидоpас!
    Дед сидит офигевший. Проходит минута:
    — Дед, скажи пять!
    — Пять…
    — Пидоpас опять!!
    Дед начинает злится… Еще через некоторое время:
    — Дед, скажи семь!
    — Ну семь…
    — Дед пидорас ты совсем!!!
    Дед теряет терпение… Раздумывает, что б такого сказать:
    — Девочка, скажи двадцать
    -Ну двадцать
    -Пошла нахуй

    Анекдот № 2:
    Профессор, устaв вытягивaть студентa нa тройку, говорит:
    - Ну лaдно … Скaжи, по кaкому предмету читaлись лекции?
    Студент молчит.
    - Тaк… Скaжи хоть, кто читaл лекции?
    Студент молчит.
    - Нaводящий вопрос: ты или я?

    Анекдот № 3:
    Стоят как-то раз американец, еврей и русский у края обрыва и спорят, у кого эхо дольше будет звучать.
    Американец изо всей силы кричит:
    -АМЕРИКААА!!!!
    Эхо звучит 10 секунд.
    Еврей подходит к краю обрыва, набирает побольше воздуха и кричит:
    -ШЕКЕЛИИИИ!!!
    Эхо звучит 20 секунд.
    Русский подходит к краю обрыва и шёпотом говорит:
    - Хохлы...
    Эхо ржёт уже третьи сутки

    Анекдот № 4:
    Один человек спросил у Сократа:
    — Знаешь, что мне сказал о тебе твой друг?
    — Подожди, — остановил его Сократ, — просей сначала то, что собираешься сказать, через три сита.
    — Он сказал, что ты пидор!
    — Прежде чем что–нибудь говорить, нужно это трижды просеять. Сначала через сито правды.
    — Что ты несёшь? Так ты пидор?
    — Значит, ты не знаешь, это правда или нет. Тогда просеем через второе сито — сито доброты. Ты хочешь сказать о моем друге что–то хорошее?
    — Бля.. ты даже разговариваешь как пидор!
    — Значит, ты собираешься сказать обо мне что–то плохое, но даже не уверен в том, что это правда. Третье сито — сито пользы. Необходимо мне услышать то, что ты хочешь рассказать?
    — Хмм… нет.
    — Пидора ответ! — заключил Сократ.

    Анекдот № 5:
    Экзамен по общей физике. Студент, который явно не подготовился, выпрашивает тройку:
    – Ну пожалуйста, Виктор Сергеевич, ну спросите еще один дополнительный вопрос!
    – Вы же совершенно ничего не знаете! Идите на пересдачу.
    – Виктор Сергеевич, нельзя мне на пересдачу, у меня еще два долга, смилуйтесь!
    – Молодой человек, имейте совесть. Увидимся осенью
    – Ну товарищ лектор! Один вопросик!
    – Ну ладно. Вот скажи мне, Писюлькин, свет - это волна?
    – Ну... волна...
    – Хорошо. А вот электрон - это волна?
    – Эээ.. ну и частица, и волна, да
    – А вот, например, протон - волна?
    – Ну если вспомнить Де Бройля...
    – Не надо никого вспоминать. Протон - волна?
    – Ну да, волна
    – Т.е. если и электрон и протон, и, скорее всего, нейтрон - это волна, то вот ваша зачетка тоже волна?
    – Получается так
    – И я - волна?
    – И вы - волна
    – Так вот везде волна а тебе на ебло говна, пиздуй отсюдова на пересдачу!


"""

import pymysql
from flask import Flask, render_template, request, redirect
from pymysql.cursors import DictCursor

app = Flask(__name__)

dbh = pymysql.connect(
    host='185.12.94.106',
    user='2p1s23',
    password='941-961-748',
    db='2p1s23',
    charset='utf8mb4',
    cursorclass=DictCursor,
    autocommit=True
)


@app.route('/login/', methods=['post', 'get'])
def login():
    message = 'Для входа введите логин и пароль'
    login, password = request.form.get('username'), request.form.get('password')  # запрос к данным формы

    if login is not None or password is not None:
        cur = dbh.cursor()
        cur.execute(f'SELECT login, password FROM users WHERE login = "{login}" AND password = "{password}";')
        a = cur.fetchone()

        if a and a['login'] == login and a['password'] == password:
            return redirect("/neyronka")
        else:
            message = "Wrong username or password"

    return render_template('login.html', message=message)


@app.route('/reg/', methods=['post', 'get'])
def registration():
    message = 'Для регистрации заполните форму ниже'
    login, password, username = request.form.get('login'), request.form.get('password'), request.form.get(
        'username')  # запрос к данным формы

    if login and password and username and len(login) >= 4 and len(password) >= 4 and len(username) >= 4:
        cur = dbh.cursor()
        cur.execute(
            f'SELECT login, password, name FROM users WHERE login = "{login}" OR password = "{password}" OR name = "{username}";')
        b = cur.fetchone()

        if b is None:
            cur.execute(
                f'INSERT INTO users (id, login, password, name) VALUES (NULL, "{login}", "{password}", "{username}");')
            return redirect("/login")

        else:
            message = "Логин, пароль или имя уже используются, попробуйте другие"

    return render_template('reg.html', message=message)


@app.route('/neyronka/', methods=['post', 'get'])
def huy():
    return render_template('neyronka.html')


@app.route('/user/<string:name>/<int:id>')
def user(name, id):
    return "User page: " + name + " - " + str(id)


if __name__ == "__main__":
    app.run(debug=True)
