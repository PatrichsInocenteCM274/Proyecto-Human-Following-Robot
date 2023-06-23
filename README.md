
# ✨ Human Following Robot en Webots 🏃‍♀️🎥🤖 ✨

```diff

+ Deep Reinforcement Learning | + DDPG | + Computer Vision | + YOLO | + Webots | + Mobile robot | 
```


Este proyecto tuvo como objetivo implementar un entorno de Gimnasio en Webots (Simulador de Físicas para Robótica). La finalidad del proyecto se enfoca en construir y entrenar un Robot movil de configuración diferencial en la tarea de supervisión o seguimiento de personas, y cumplir este objetivo evitando colisionar con elementos del entorno que puedan poner en peligro al robot.  El Robot Móvil cuenta con una cámara RGB, un sensor LIDAR, dos actuadores que controlan el giro de las 2 ruedas que nueven al robot, un actuador que controla la dirección de visión de la cámara y 3 sensores (de rotación y velocidad de los diversos motores del robot). Específicamente, el robot deberá moverse en dirección de una persona virtual escogida a priori, y cuya inteligencia de toma de decisiones estará dada por la conjunción de dos modelos, el primero, es un modelo CNN muy ligero llamado Tiny-Yolo v3 que realizará la detección de la persona virtual mediante las imagenes capturadas por la cámara y el segundo, es un modelo de Aprendizaje por Refuerzo profundo llamado DDPG, el cual actuará como el controlador de locomoción del robot.

### <ins>Analisis de Entorno</ins>
En este proyecto se modelo un Entorno parcialmente observable por el agente robótico móvil, donde en cada instante de tiempo, el agente registrará los datos de observación de 64 mediciones de puntos del LIDAR, los datos de los sensores de orientación, rotación y velocidad, es importante mencionar que la orientacion de cámara actual es enviada también como observación. La orientación de la cámara dependerá de la dirección en la que la persona objetivo es detectada por el modelo YOLO. A continuación se muestra un gráfico explicativo del entorno:

![boceto](https://github.com/PatrichsInocenteCM274/Proyecto-Human-Following-Robot/assets/30361234/0412d654-5be3-418e-889a-eb661d192b56)

### <ins>El Robot</ins>
Desde el inicio, pensé en una configuración diferencial para mi robot, por la simplicidad de su locomoción, ya que solo depende de dos actuadores o motores por lo que ayuda a reducir la complejidad de acciones al momento de la busqueda de la política optima en nuestro modelo DDPG, principalmente en Webots existen algunas opciones útiles de robots conocidos, uno de los robots que más llamo mi atención fue el robot Thiago Base de la empresa Pal Robotics que posee una configuración diferencial por ello partiendo de su modelo, elimine, modifique y añadi partes adicionales para terminar implementando mi robot personalizado, por ejemplo, mejores ruedas locas base, un torso con un sensor lidar centrado y una base giratoria como cabeza del robot, la cual ayuda a mover la cámara desde un angulo entre 0 a 360 grados, A contiación el resultado de esta construcción se muestra:

![robot_demo](https://github.com/PatrichsInocenteCM274/Proyecto-Human-Following-Robot/assets/30361234/84b51296-d825-4abf-9f69-5eb95130d499)
