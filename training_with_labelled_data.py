# Training based on labelled data
# epochs = 30
# for epoch in range(epochs):
#     model.train()
#     total_loss = 0
#     for batch in train_loader:
#         batch = batch.to(device)
#         optimizer.zero_grad()
#         out = model(batch.x, batch.edge_index, batch.batch)
#         # For now use random dummy targets
#         target = torch.randint(0, 2, (out.size(0),), device=device)
#         loss = criterion(out, target)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     print(f"Epoch {epoch:02d} | Loss: {total_loss:.4f}")
