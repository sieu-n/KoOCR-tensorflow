def predict_complete(model,image,n=1):
        #Predict the top-n classes of the image
        #Returns top n characters that maximize the probability
        charset=korean_manager.load_charset()
        if image.shape==(256,256):
            image=image.reshape((1,256,256))
        pred_class=model.predict(image)
        pred_class=np.argsort(pred_class,axis=1)[:,-n:]
        pred_hangeul=[]
        for idx in range(image.shape[0]):
            pred_hangeul.append([])
            for char in pred_class[idx]:
                pred_hangeul[-1].append(charset[char])
        return pred_hangeul

def split_topn(cho_pred,jung_pred,jong_pred,n):
    cho_idx,jung_idx,jong_idx=np.argsort(cho_pred,axis=1)[:,-k:],np.argsort(jung_pred,axis=1)[:,-k:],np.argsort(jong_pred,axis=1)[:,-k:]
    cho_pred,jung_pred,jong_pred=np.sort(cho_pred,axis=1)[:,-k:],np.sort(jung_pred,axis=1)[:,-k:],np.sort(jong_pred,axis=1)[:,-k:]
    #Convert indicies to korean character
    pred_hangeul=[]
    for idx in range(image.shape[0]):
        pred_hangeul.append([])

        cho_prob,jung_prob,jong_prob=cho_pred[idx],jung_pred[idx].reshape(-1,1),jong_pred[idx].reshape(-1,1)

        mult=((cho_prob*jung_prob).flatten()*jong_prob).flatten().argsort()[-5:][::-1]
        for max_idx in mult:
            pred_hangeul[-1].append(korean_manager.index_to_korean((cho_idx[idx][max_idx%k],jung_idx[idx][(max_idx%(k*k))//k]\
                ,jong_idx[idx][max_idx//(k*k)])))

    return pred_hangeul

def predict_ir(model,image,n=1, t=4):
    def find_target_layer(model):
		# attempt to find the final convolutional layer in the network
		# by looping over the layers of the network in reverse order
		for layer in reversed(model.layers):
			# check to see if the layer has a 4D output
			if len(layer.output_shape) == 4:
				return layer.name
		# otherwise, we could not find a 4D layer so the GradCAM
		# algorithm cannot be applied
		raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")
    return 0

def predict_split(model,image,n=1):
    #Predict the top-n classes of the image
    #k: top classes for each component to generate
    #Returns top n characters that maximize pred(chosung)*pred(jungsung)*pred(jongsung)
    k=int(n**(1/3))+2
    if image.shape==(256,256):
        image=image.reshape((1,256,256))
    #Predict top n classes
    
    prediction_list=model.predict(image)
    prediction_dict = {name: pred for name, pred in zip(model.output_names, prediction_list)}
    cho_pred,jung_pred,jong_pred=prediction_dict['CHOSUNG'],prediction_dict['JUNGSUNG'],prediction_dict['JONGSUNG']
    return split_topn(cho_pred,jung_pred,jong_pred)