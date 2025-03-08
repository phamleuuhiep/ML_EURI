import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from models.ga_cnn import SimpleCNN
import torch.optim as optim
import torch.nn as nn

filters_list = [16, 32, 64, 128]
dense_units_list = [32, 64, 128]
lr_list = [1e-4, 1e-3, 1e-2, 1e-1]
batch_size_list = [16, 32, 64, 128]
dropout_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

def initialize_chromosome():
    return {
        'filters': random.choice(filters_list),
        'dense_units': random.choice(dense_units_list),
        'lr': random.choice(lr_list),
        'batch_size': random.choice(batch_size_list),
        'dropout': random.choice(dropout_list)
    }

def crossover(parent1, parent2):
    return {key: random.choice([parent1[key], parent2[key]]) for key in parent1}

def mutate(chromosome, mutation_rate=0.1):
    mutated_chromosome = chromosome.copy()
    for key in mutated_chromosome:
        if random.random() < mutation_rate:
            if key == 'filters':
                mutated_chromosome[key] = random.choice(filters_list)
            elif key == 'dense_units':
                mutated_chromosome[key] = random.choice(dense_units_list)
            elif key == 'lr':
                mutated_chromosome[key] = random.choice(lr_list)
            elif key == 'batch_size':
                mutated_chromosome[key] = random.choice(batch_size_list)
            elif key == 'dropout':
                mutated_chromosome[key] = random.choice(dropout_list)
    return mutated_chromosome

def evaluate_model(hparams, train_dataset, test_dataset, epochs=5):
    train_loader = DataLoader(train_dataset, batch_size=hparams['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=hparams['batch_size'], shuffle=False)
    model = SimpleCNN(filters=hparams['filters'], dense_units=hparams['dense_units'], num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=hparams['lr'])
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def genetic_algorithm(train_dataset, test_dataset, population_size=10, generations=5, mutation_rate=0.1):
    target_score = 0.95
    best_scores = []
    mean_scores = []
    population = [initialize_chromosome() for _ in range(population_size)]
    for generation in range(generations):
        print(f"Generation {generation + 1}")
        scores = []
        for chromosome in population:
            accuracy = evaluate_model(chromosome, train_dataset, test_dataset, epochs=5)
            scores.append(accuracy)
            print(f"Chromosome: {chromosome}, Accuracy: {accuracy:.2f}%")
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top_chromosomes = [population[i] for i in sorted_indices[:population_size // 2]]
        new_population = top_chromosomes.copy()
        while len(new_population) < population_size:
            parent1, parent2 = random.choices(top_chromosomes, k=2)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            new_population.append(child)
        best_scores.append(max(scores))
        mean_scores.append(np.mean(scores))
        population = new_population
    return population[sorted_indices[0]]