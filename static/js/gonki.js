const p1 = {left:0, right:0, up:0, speed:0, tl:null},
    p2 = {...p1}

gsap.set('.car1, .car2', {scaleX:-0.15, scaleY:0.15})

p1.tl = gsap.timeline({repeat:-1})
    .to('.car1Group', {ease:'none',
        motionPath:{
            path: '#track',
            align: '#track',
            alignOrigin: [0.5, 0.5],
            autoRotate: true
        }
    })
    .pause(0.01)

p2.tl = gsap.timeline({repeat:-1})
    .to('.car2Group', {ease:'none',
        motionPath:{
            path: '#track',
            align: '#track',
            alignOrigin: [0.5, 0.5],
            autoRotate: true
        }
    })
    .pause(0.01)


gsap.set('.car1', {y:-7})
gsap.set('.car2', {y:7})
gsap.to('svg', {opacity:1, ease:'power2.inOut'})

$(window).on('keydown', (e)=>{ //console.log(e.keyCode)
    e.preventDefault();
    if (e.keyCode==65) p1.left = 1;
    if (e.keyCode==87) p1.up = 1;
    if (e.keyCode==68) p1.right = 1;
    if (e.keyCode==37) p2.left = 1;
    if (e.keyCode==38) p2.up = 1;
    if (e.keyCode==39) p2.right = 1;
});

$(window).on('keyup', (e)=>{
    if (e.keyCode==65) p1.left = 0;
    if (e.keyCode==87) p1.up = 0;
    if (e.keyCode==68) p1.right = 0;
    if (e.keyCode==37) p2.left = 0;
    if (e.keyCode==38) p2.up = 0;
    if (e.keyCode==39) p2.right = 0;
});

gsap.ticker.add((t,d,f)=>{

    const car1Pos = {x:gsap.getProperty('.car1Group', 'x'), y:gsap.getProperty('.car1Group', 'y')};
    gsap.to('#pov1', {x:-car1Pos.x+99, y:-car1Pos.y+85, duration:0.1})
    gsap.to('.leftSide .main', {attr:{viewBox:(45*p1.speed*100)+' '+(45*p1.speed*100)+' '+(165-45*p1.speed*100)+' '+(165-45*p1.speed*100)}})

    const car2Pos = {x:gsap.getProperty('.car2Group', 'x'), y:gsap.getProperty('.car2Group', 'y')};
    gsap.to('#pov2', {x:-car2Pos.x+99, y:-car2Pos.y+85, duration:0.1})
    gsap.to('.rightSide .main', {attr:{viewBox:(45*p2.speed*100)+' '+(45*p2.speed*100)+' '+(165-45*p2.speed*100)+' '+(165-45*p2.speed*100)}})

    let car1Slow = 1
    const car1_y = gsap.getProperty('.car1', 'y');
    if (p1.right && car1_y<15) gsap.to('.car1', {y:'+='+250*p1.speed})
    if (p1.left && car1_y>-15) gsap.to('.car1', {y:'-='+250*p1.speed})
    if (car1_y>12||car1_y<-12) car1Slow = 0.5
    if (car1_y>15||car1_y<-15) car1Slow = 0.3
    gsap.to('.car1', {duration:1, rotate:(20*(p1.right+-p1.left))*(p1.speed*120)})

    if (p1.up && p1.speed<0.013*car1Slow) gsap.to(p1, {speed:'+=0.000'+7*car1Slow})
    else gsap.to(p1, {speed:0, duration:3})

    let car2Slow = 1
    const car2_y = gsap.getProperty('.car2', 'y');
    if (p2.right && car2_y<15) gsap.to('.car2', {y:'+='+250*p2.speed})
    if (p2.left && car2_y>-15) gsap.to('.car2', {y:'-='+250*p2.speed})
    if (car2_y>12||car2_y<-12) car2Slow = 0.5
    if (car2_y>15||car2_y<-15) car2Slow = 0.3
    gsap.to('.car2', {duration:1, rotate:(20*(p2.right+-p2.left))*(p2.speed*120)})

    if (p2.up && p2.speed<0.013*car2Slow) gsap.to(p2, {speed:'+=0.000'+7*car2Slow})
    else gsap.to(p2, {speed:0, duration:3})

    if (f%9==0) gsap.set('.mph', {textContent:(i)=>eval('p'+(i+1)).speed*11300, snap:'textContent', ease:'none'})
    gsap.to(p1.tl, {progress:'+='+p1.speed})
    gsap.to(p2.tl, {progress:'+='+p2.speed})
})