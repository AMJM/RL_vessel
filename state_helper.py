'''
[AKUM-12/12/2018]
Codigo legado do Data Filter adpatados (P3D_file, Point e Ship)
Data Filter: projeto para gerar os dados utilizados na simulacao

Proposito: codigo auxiliar para modelagem interna dos dados e compatibilizar a comunicacao entre os dados do Dyna e da rede neural
'''


'''
Importacao das bibliotecas a serem utilizadas
'''
import math as m
import re
from simulation_settings import *


'''
Classe para suporte de arquivos .p3d
'''
class P3D_file:
    # Construtor da classe
    # Entrada:
    #   path: string com o diretorio do arquivo p3d/nome do arquivo.p3d
    # Saida:
    #   None
    def __init__(self, path):
        self.path = path

    # Metodo para extracao das dimensoes da embarcao do arquivo
    # Entrada:
    #   None
    # Saida:
    #   beam: largura da embarcacao
    #   height: altura da embarcacao
    #   length: comprimento da embarcacao
    def find_dimensions(self):
        # Uso de regex para padrao dos parametros da embarcacao no arquivo p3d
        regex = re.compile("VESSEL[\s\S\n]*?(?<=BEAM = )(\d+.\d+)[\s\S\n]*?(?<=HEIGHT = )(\d+.\d+)[\s\S\n]*?(?<=LENGTH = )(\d+.\d+)")
        p3d_file = open(self.path)
        dim = regex.findall(p3d_file.read())

        if len(dim[0]) != 3:  # Padrao de regex nao encontrado
            print("ERRO: Dados da embarcacao nao encontrados!")
            return [-1, -1, -1]
        else:  # Separacao dos dados encontrados
            beam = float(dim[0][0])
            height = float(dim[0][1])
            length = float(dim[0][2])
            return beam, height, length


'''
Classe para definicao de um ponto ou vetor em coordenadas cartesianas
'''
class Point:
    # Construtor da classe
    # Entrada:
    #   x: coordenada x do ponto
    #   y: coordenada y do ponto
    # Saida:
    #   None
    def __init__(self, x, y, zz=None):
        self.x = x
        self.y = y
        self.zz = zz
        self.mod = m.sqrt(x ** 2 + y ** 2)
        self.angle = m.degrees(m.atan2(y, x))


'''
Classe para suporte de embarcacoes
'''
class Ship:
    # Construtor da classe
    # Entrada:
    #   dim: vetor com as dimensoes da embarcacao - [beam height length]
    # Saida:
    #   None
    def __init__(self, dim):
        self.beam = dim[0]
        self.height = dim[1]
        self.length = dim[2]

    # Metodo para calculo de distancias da embarcacao ate as margens e target
    # Entrada:
    #   center: coordenada cartesiana do centro da embarcacao
    #   angle: angulo de aproamento da embarcacao em graus
    #   buoys: vetor com as posicoes das boias aos pares
    #   target: coordenada cartesiana do target
    # Saida:
    #   dpb: distancia a bombordo
    #   dsb: distancia a borest
    #   dtg: distancia ao target
    def calc_dist_lateral(self, center, angle, buoys, target):
        # Coordenadas medias frontal e traseira
        front = Point(center.x + self.length / 2 * m.cos(m.radians(angle)), center.y + self.beam / 2 * m.sin(m.radians(angle)))
        back = Point(center.x - self.length / 2 * m.cos(m.radians(angle)), center.y - self.beam / 2 * m.sin(m.radians(angle)))

        # Coordenadas dos vertices - considera a embarcacao um retangulo
        front_sb = Point(center.x + self.length / 2 * m.cos(m.radians(angle)) + self.beam / 2 * m.sin(m.radians(angle)), center.y + self.length / 2 * m.sin(m.radians(angle)) - self.beam / 2 * m.cos(m.radians(angle)))
        front_pb = Point(center.x + self.length / 2 * m.cos(m.radians(angle)) - self.beam / 2 * m.sin(m.radians(angle)), center.y + self.length / 2 * m.sin(m.radians(angle)) + self.beam / 2 * m.cos(m.radians(angle)))
        back_sb = Point(center.x - self.length / 2 * m.cos(m.radians(angle)) + self.beam / 2 * m.sin(m.radians(angle)), center.y - self.length / 2 * m.sin(m.radians(angle)) - self.beam / 2 * m.cos(m.radians(angle)))
        back_pb = Point(center.x - self.length / 2 * m.cos(m.radians(angle)) - self.beam / 2 * m.sin(m.radians(angle)), center.y - self.length / 2 * m.sin(m.radians(angle)) + self.beam / 2 * m.cos(m.radians(angle)))

        # Determina em qual secao de boias esta o ponto medio frontal e traseiro
        section_front = self._determine_section(front, buoys)
        section_back = self._determine_section(back, buoys)

        # Determina a direcao da embarcacao em relacao a linha media
        direction = self._determine_direction(section_front, angle, buoys)

        if direction == -1:  # virado a bombordo
            sh_pb = front_pb
            sh_sb = back_sb
            section_pb = section_front
            section_sb = section_back
        else:  # na direcao ou virado a estibordo
            sh_pb = back_pb
            sh_sb = front_sb
            section_pb = section_back
            section_sb = section_front
        dsb = self._dist_line_point(buoys[section_sb], buoys[section_sb + 2], sh_sb, -1)  # distancia estibordo
        dpb = self._dist_line_point(buoys[section_pb + 1], buoys[section_pb + 3], sh_pb, 1)  # distancia bombordo
        dtg = self._dist_point_point(center, target)  # distancia target

        # Embarcacao contrario a entrada no canal
        if ~((angle % 360 > 135) and (angle % 360 < 315)):
            print("ERRO: Direcao da embarcacao em saida!")

        return [dpb, dsb, dtg]

    # Metodo para calculo de distancias da embarcacao ate a linha central e target
    # Entrada:
    #   center: coordenada cartesiana do centro da embarcacao
    #   angle: angulo de aproamento da embarcacao em graus
    #   buoys: vetor com as posicoes das boias aos pares
    #   target: coordenada cartesiana do target
    # Saida:
    #   dml: distancia a bombordo
    #   dtg: distancia ao target
    def calc_dist_midline(self, center, angle, buoys, target):
        # Determina em qual secao de boias esta o ponto medio frontal e traseiro
        section = self._determine_section(center, buoys)

        b1 = buoys[section]
        b2 = buoys[section + 1]
        p1 = Point((b1.x + b2.x)/2, (b1.y + b2.y)/2)

        b1 = buoys[section + 2]
        b2 = buoys[section + 3]
        p2 = Point((b1.x + b2.x) / 2, (b1.y + b2.y) / 2)

        dml = self._dist_line_point(p1, p2, center, 1)  # distancia central
        dtg = self._dist_point_point(center, target)  # distancia target

        # Embarcacao contrario a entrada no canal
        if ~((angle % 360 > 135) and (angle % 360 < 315)):
            print("ERRO: Direcao da embarcacao em saida!")

        return [dml, dtg]

    # Metodo para calculo de cog (course over ground) e sog (speed over ground)
    # Entrada:
    #   center: coordenada cartesiana do centro da embarcacao
    #   angle: angulo de aproamento da embarcacao em graus
    #   buoys: vetor com as posicoes das boias aos pares
    #   vx: velocidade surge (direção longitudinal da embarcação)
    #   vy: velocidade sway (direção no eixo lateral da embarcação)
    # Saida:
    #   cog: angulo da velocidade total
    #   sog: modulo da velocidade total
    def calc_cog_sog(self, center, angle, buoys, vx, vy):
        # Determina em qual secao de boias esta o ponto medio frontal e traseiro
        section = self._determine_section(center, buoys)

        b1 = buoys[section]
        b2 = buoys[section + 1]
        p1 = Point((b1.x + b2.x) / 2, (b1.y + b2.y) / 2)

        b1 = buoys[section + 2]
        b2 = buoys[section + 3]
        p2 = Point((b1.x + b2.x) / 2, (b1.y + b2.y) / 2)

        # Define um vetor velocidade
        vec_vel = Point(vx, vy)

        # Define o angulo da linha media do canal
        channel_angle = self._angle_point_point(p1, p2)

        # cog global = aproamento + angulo da velocidade
        cog = angle + vec_vel.angle
        # sog = modulo da velocidade total
        sog = vec_vel.mod
        # cog local = deslocamento do cog em relacao ao angulo do canal
        local_cog = channel_angle - cog

        return [cog, sog, local_cog]

    # Metodo para definicao entre quais boias esta a embarcacao
    # Entrada:
    #   point: coordenada cartesiana do ponto a ser verificado
    #   buoys: vetor com as posicoes das boias
    # Saida:
    #   Valor que representa qual porcao do canal esta com referencia da entrada para a saida
    def _determine_section(self, point, buoys):
        for i in range(0, len(buoys), 2):
            # Verifica se passou de cada limite de boia
            if point.x < (buoys[len(buoys) - i - 1].x + buoys[len(buoys) - i - 2].x) / 2:
                if len(buoys) - i - 2 == len(buoys) - 2:
                    return len(buoys) - i - 4  # Desconsidera se passa do target
                else:
                    return len(buoys) - i - 2
        return len(buoys) - i - 2

    # Metodo para definicao da direcao da embarcacao em relacao a linha media
    # Entrada:
    #   section_front: porcao do canal em que a embarcacao esta
    #   angle: angulo de aproamento em graus
    #   buoys: vetor com as posicoes das boias
    # Saida:
    #   -1 para bombordo, 0 na mesma direcao e 1 para estibordo
    def _determine_direction(self, section_front, angle, buoys):
        # Define a direcao da linha media
        ini = Point((buoys[section_front].x + buoys[section_front + 1].x) / 2, (buoys[section_front].y + buoys[section_front + 1].y) / 2)
        end = Point((buoys[section_front + 2].x + buoys[section_front + 3].x) / 2, (buoys[section_front + 2].y + buoys[section_front + 3].y) / 2)
        line_angle = m.degrees(m.atan((end.y - ini.y) / (end.x - ini.x))) - 180

        # Compara a embarcacao com a linha media
        if line_angle - angle > 0:
            return 1
        elif line_angle - angle == 0:
            return 0
        else:
            return -1

    # Metodo para calculo de distancia entre linha e ponto
    # Entrada:
    #   line_p_1: coordenadas do primeiro ponto que define a linha
    #   line_p_1: coordenadas do segundo ponto que define a linha
    #   point: coordenadas do ponto de interesse
    #   type: -1 para relacao estibordo e 1 para relacao bombordo
    # Saida:
    #   Distancia do ponto a linha
    def _dist_line_point(self, line_p_1, line_p_2, point, type):
        # Define a linha por dois pontos
        y_diff = line_p_2.y - line_p_1.y
        x_diff = line_p_2.x - line_p_1.x
        y_line = y_diff / x_diff * (point.x - line_p_1.x) + line_p_1.y

        # Fator para verificar se passou do canal (linha)
        factor = 1
        if (y_line > point.y and type == 1) or (y_line < point.y and type == -1):
            factor = -1

        return factor * abs(y_diff * point.x - x_diff * point.y + line_p_2.x * line_p_1.y - line_p_2.y * line_p_1.x) / m.sqrt(y_diff ** 2 + x_diff ** 2)

    # Metodo para calculo de distancia entre linha e ponto
    # Entrada:
    #   p1: coordenadas do primeiro ponto
    #   p2: coordenadas do segundo ponto
    # Saida:
    #   Distancia entre os pontos
    def _dist_point_point(self, p1, p2):
        return m.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

    # Metodo para calculo de angulo entre pontos
    # Entrada:
    #   p1: coordenadas do primeiro ponto
    #   p2: coordenadas do segundo ponto
    # Saida:
    #   Angulo entre pontos
    def _angle_point_point(self, p1, p2):
        return m.degrees(m.atan2(p2.y - p1.y, p2.x - p1.x))


'''
Classe principal auxiliar da interface Dyna e rede neural
Obs.: dependendo do modelo de rede neural, sera muito importante alterar os metodos get_inputtable_rudder_state e get_inputtable_prop_state para compatibilizar a comunicacao
'''
class State_Helper:
    # Construtor da classe
    # Entrada:
    #   path: endereco para o arquivo p3d
    #   buoys: vetor com as posicoes das boias
    #   target: coordenada cartesiana do target
    # Saida:
    #   None
    def __init__(self, path, buoys, target):
        ship_dim = P3D_file(path).find_dimensions()
        self.ship = Ship(ship_dim)
        self.buoys = buoys
        self.target = target

    # Gera informacoes auxiliares sobre os estados - distancia bombordo, boreste, cog, sog e local sog
    # Entrada:
    #   state: vetor com o estado [posicao_x, posicao_y, angulo_aproamento, velocidade_avanco, velocidade_deriva, velocidade_guinada]
    # Saida:
    #   Vetor dos estados auxiliares [distancia_bombordo, distancia_boreste, course_over_ground, speed_over_ground, course_over_ground_local]
    def get_aux_states(self, state):
        from simulation_settings import ST_MID, ST_TARGET, ST_POSX, ST_POSY, ST_POSZZ, ST_VELX, ST_VELY, ST_VELZZ, ST_DPB, ST_DSB, ST_COG, ST_SOG, ST_LOCAL_COG
        center = Point(state[ST_POSX], state[ST_POSY])
        angle = self.get_converted_angle(state[ST_POSZZ])
        position = self.ship.calc_dist_lateral(center, angle, self.buoys, self.target)
        velocity = self.ship.calc_cog_sog(center, angle, self.buoys, state[ST_VELX], state[ST_VELY])
        return [position[ST_DPB], position[ST_DSB], velocity[ST_COG], velocity[ST_SOG], velocity[ST_LOCAL_COG]]

    # Gera o estado no formato esperado pela rede neural de controle do leme
    # Entrada:
    #   state: vetor com o estado [posicao_x, posicao_y, angulo_aproamento, velocidade_avanco, velocidade_deriva, velocidade_guinada]
    # Saida:
    #   Vetor dos estados auxiliares [angulo_aproamento, velocidade_avanco, velocidade_deriva, velocidade_guinada, distancia_linha_media]
    def get_inputtable_rudder_state(self, state):
        from simulation_settings import ST_MID, ST_TARGET, ST_POSX, ST_POSY, ST_POSZZ, ST_VELX, ST_VELY, ST_VELZZ
        center = Point(state[ST_POSX], state[ST_POSY])
        angle = self.get_converted_angle(state[ST_POSZZ])
        position = self.ship.calc_dist_midline(center, angle, self.buoys, self.target)
        # Aparentemente a base utilizada e negativa entao para compatibilizar precisa do oposto da velocidade de guinada
        return [angle, state[ST_VELX], state[ST_VELY], -state[ST_VELZZ], position[ST_MID]]

    # Gera o estado no formato esperado pela rede neural de controle de maquina
    # Entrada:
    #   state: vetor com o estado [posicao_x, posicao_y, angulo_aproamento, velocidade_avanco, velocidade_deriva, velocidade_guinada]
    # Saida:
    #   Vetor dos estados auxiliares [angulo_aproamento, velocidade_avanco, velocidade_deriva, velocidade_guinada, distancia_linha_media, distancia_target]
    def get_inputtable_prop_state(self, state):
        from simulation_settings import ST_MID, ST_TARGET, ST_POSX, ST_POSY, ST_POSZZ, ST_VELX, ST_VELY, ST_VELZZ
        center = Point(state[ST_POSX], state[ST_POSY])
        angle = self.get_converted_angle(state[ST_POSZZ])
        position = self.ship.calc_dist_midline(center, angle, self.buoys, self.target)
        # Aparentemente a base utilizada e negativa entao para compatibilizar precisa do oposto da velocidade de guinada
        return [angle, state[ST_VELX], state[ST_VELY], -state[ST_VELZZ], position[ST_MID], position[ST_TARGET]]

    # Gera a ordem de maquina no formato esperado pelo Dyna (percentual da rotacao total)
    # Entrada:
    #   action: ordem de maquina gerada pela rede neural
    # Saida:
    #   Ordem de maquina esperada pelo Dyna
    def get_converted_propeller(self, action):
        from simulation_settings import discrete_velocity, NUM_PROP_VEL
        return discrete_velocity[action + NUM_PROP_VEL]

    # Gera a ordem de leme no formato esperado pelo Dyna (percentual do leme maximo)
    # Entrada:
    #   action: ordem de leme gerado pela rede neural
    # Saida:
    #   Ordem de leme esperado pelo Dyna
    def get_converted_rudder(self, action):
        from simulation_settings import MAX_RUDDER_RAD
        return action / MAX_RUDDER_RAD

    # Converte angulo entre coordenada cartesiana (leste-antihorario) e coordenada maritima (norte-horaria)
    # Entrada:
    #   angle: angulo em graus
    # Saida:
    #   Angulo em graus convertido
    def get_converted_angle(self, angle):
        return -270 - angle

    # Posiciona a embarcacao no mapeamento do ambiente
    # Entrada:
    #   state: vetor com o estado [posicao_x, posicao_y, angulo_aproamento, velocidade_avanco, velocidade_deriva, velocidade_guinada]
    # Saida:
    #   Angulo em graus convertido
    def set_position(self, state):
        from simulation_settings import ST_POSX, ST_POSY, ST_POSZZ, geom_helper
        geom_helper.set_polygon_position(state[ST_POSX], state[ST_POSY], state[ST_POSZZ])
        if geom_helper.ship_collided():
            print('SHIP COLLIDED!!!')

    # Converte o vetor de estado global para o local
    # Entrada:
    #   state: vetor com o estado [posicao_x, posicao_y, angulo_aproamento, velocidade_x_global, velocidade_y_global, velocidade_guinada]
    # Saida:
    #   Vetor dos estados locais [posicao_x, posicao_y, angulo_aproamento, velocidade_avanco, velocidade_deriva, velocidade_guinada]
    def convert_cartesian_to_local(self, state):
        from simulation_settings import ST_POSX, ST_POSY, ST_POSZZ, ST_VELX, ST_VELY, ST_VELZZ
        heading_e_ccw = 90 - state[ST_POSZZ]
        theta = np.radians(heading_e_ccw)
        c, s = np.cos(theta), np.sin(theta)
        # Matriz de rotacao para os vetores de velocidade
        m = np.matrix(((c, s), (-s, c)))
        global_array = np.matrix([[state[ST_VELX], state[ST_VELY]]]).T
        transformed = m * global_array
        return state[ST_POSX], state[ST_POSY], state[ST_POSZZ], transformed.item(0), transformed.item(1), state[ST_VELZZ]