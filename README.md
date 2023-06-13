
# ✨ Human Following Robot en Webots 🏃‍♀️🎥🤖 ✨

```diff

+ Deep Reinforcement Learning | + DDPG | + Computer Vision | + YOLO | + Webots | + Mobile robot | 
```

Este proyecto tuvo como objetivo implementar un entorno de Gimnasio en Webots con la finalidad de entrenar un Robot movil a seguir a una persona objetivo dentro del escenario simulado evitando obstaculos. El Robot Móvil cuenta con una cámara, un sensor LIDAR, dos actuadores que controlan el giro de las 2 ruedas que nueven al robot, un actuador que controla el giro de la cámara y 3 sensores de rotación y velocidad para los actuadores del robot. 

### <ins>Analisis de Entorno</ins>
En este proyecto se modelo un Entorno parcialmente observable por el agente robótico móvil, donde en cada instante de tiempo, el agente registrará los datos de observación de 64 mediciones de puntos del LIDAR, los datos de los sensores de orientación, rotación y velocidad, es importante mencionar que la orientacion de cámara actual es enviada también como observación. La orientación de la cámara dependerá de la dirección en la que la persona objetivo es detectada por el modelo YOLO. A continuación se muestra un gráfico explicativo del entorno:

![analisis_entorno](https://github.com/PatrichsInocenteCM274/Proyecto-Human-Following-Robot/assets/30361234/5feaa831-8b04-47f4-a1db-59e19560259b)
