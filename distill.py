import torch

def distill_one_epoch(student,teacher,loader,criterion,optimizer,device):
	# Student and teacher have same embedding dim (feature map size). 
	# criterion would be the L2 loss or cosine similarity between the embeddings. 
	# Student learns teacher embeddings.
	student.train()
	running_loss=0.0
	for inputs,_ in loader:
		inputs=inputs.to(device)
		optimizer.zero_grad()
		student_outputs=student(inputs)
		teacher_outputs=teacher(inputs)
		loss=criterion(student_outputs,teacher_outputs)
		loss.backward()
		optimizer.step()
		running_loss+=loss.item()
	return running_loss/len(loader.dataset)

def distill(student,teacher,data_loader,optimizer,device,num_epochs=1):
	embedding_dim=student.embedding_dim # unverified code
	assert student.embedding_dim==teacher.embedding_dim # unverified code
	cosine_similarity=torch.nn.CosineSimilarity(dim=embedding_dim)
	for epoch in range(num_epochs):
		loss=distill_one_epoch(student,teacher,data_loader,cosine_similarity,optimizer,device)
		print('Epoch %d, Loss: %.4f'%(epoch,loss))
