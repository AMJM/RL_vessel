'''
[AKUM-14/12/2018]
Arquivo apenas para facilitar na chamada das simulacoes de forma padronizada

Obs1.: Nao esquecer de ajustar os parametros da simulacao em simulation_settings.py
Obs2.: Nao esquecer de abrir o chat_server
'''

import communication

if __name__ == "__main__":
    # Codigo do modelo a ser utilizado no teste
    # Necessario ter o modelo na pasta: ./keras_logs/<train_code>
    # A pasta deve conter o scaler dos dados .save e o modelo .h5
    train_code = 'model_2018-10-19_23-17'

    # Lista das posicoes de teste utilizadas para elaboracao do relatorio
    # Posicao inicial da embarcacao [x, y, zz, vx, vy, vzz] = [ST_POSX, ST_POSY, ST_POSZZ, ST_VELX, ST_VELY, ST_VELZZ]
    ## SC1
    #init_state = [11611.97, 5404.08, -166.45, 4.00, 0.00, 0.00]
    init_state = [11601.97, 5445.60, -166.45, 4.00, 0.00, 0.00]  # Posicao neutra
    #init_state = [11591.96, 5487.15, -166.45, 4.00, 0.00, 0.00]

    ## SC2
    #init_state = [11601.97, 5445.60, -151.45, 4.00, 0.00, 0.00]
    #init_state = [11601.97, 5445.60, -176.45, 4.00, 0.00, 0.00]

    ## SC3
    #init_state = [11601.97, 5445.60, -166.45, 4.00, 0.00, 0.11]
    #init_state = [11601.97, 5445.60, -166.45, 4.00, 0.00, -0.11]

    # As redes neurais recorrentes devem ser treinadas de forma stateless, mas executados de forma stateful
    # O Keras nao permite essa transicao de forma simples, por isso a necessidade de saber qual foi o modelo treinado (num_model)
    # A solucao encontrada foi ter o modelo stateless treinado salvo em .h5, recriar um novo modelo identico ao original mas stateful e transferir os pesos do modelo treinado para o novo modelo
    # Em controller.py os modelos estao definicos como model_rudder_<num_model> e model_propeller_<num_model>, basta definir o valor de num_model para que ele o codigo busque o modelo adequado
    # dict_model estao os modelos que foram utilizados para o relatorio
    dict_model = {'model_2018-10-17_14-14': 6, 'model_2018-10-18_0-58': 11, 'model_2018-10-18_11-39': 17, 'model_2018-10-19_23-17': 52}
    num_model = dict_model[train_code]

    # Faz a avaliacao do modelo e salva o resultado
    final_flag, step, dfStates, viewer = communication.evaluate_model(train_code, init_state, num_model)
    communication.write_log(train_code, init_state=init_state, flag_header=True, final_flag=final_flag, num_step=step, dfStates=dfStates, viewer=viewer)