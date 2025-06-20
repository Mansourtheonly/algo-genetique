// sensor_ga.cpp
// ------------------------------------------------------------
// Genetic Algorithm for Optimal Sensor Placement (Coverage + Connectivity)
// Author: Cascade AI assistant (2025)
// ------------------------------------------------------------
//  Dependencies:
//   • C++17 compliant compiler (e.g. g++ 11+, MSVC 2019+, clang 12+)
//   • nlohmann/json single-header library (https://github.com/nlohmann/json)
//     Place the header as "json.hpp" in the same directory or include path.
//
//  Usage:
//     sensor_ga <path-to-data.json>
//
//  The program reads the JSON input, then iterates over possible numbers of
//  sensors N (maxLocations → 1), running a GA for each N until it can no longer
//  find a valid solution.  It prints exhaustive diagnostics to both stdout and
//  "results.txt".
// ------------------------------------------------------------

#include <algorithm>
#include <cassert>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "json.hpp"  // nlohmann JSON single header

using json = nlohmann::json;

// ------------------------------------------------------------
// Basic Geometry Helpers
// ------------------------------------------------------------
struct Point {
    double x{};
    double y{};
};

double dist(const Point &a, const Point &b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

// ------------------------------------------------------------
// Data Structures
// ------------------------------------------------------------
struct GeneticParams {
    int    population_size      = 5;   // ↓ from 5000
    int    generations          = 300;   // still plenty
    int    tournament_size      = 3;
    double prob_crossover       = 0.8;
    double prob_mutation        = 0.2;
    double prob_gene_mutation   = 0.05;
    int    num_elites           = 4;     // a few more elites is safe
    int    stagnation_limit     = 60;    // ⅕ of generations
};

struct SensorSpec {
    double sense_range = 1.0; // coverage range
    double comm_range = 2.0;  // communication range
};

struct ProblemInstance {
    std::vector<Point> locations; // candidate sensor spots
    std::vector<Point> pois;      // points of interest
    SensorSpec sensor;
    GeneticParams gparams;
};

struct Individual {
    std::vector<int> genes; // indices into locations (size = N)
    double fitness = 0.0;
    bool valid = false;
};

// ------------------------------------------------------------
// Pretty Printing Helpers
// ------------------------------------------------------------
std::string genesToString(const std::vector<int> &genes) {
    std::string s = "[";
    for (size_t i = 0; i < genes.size(); ++i) {
        s += std::to_string(genes[i]);
        if (i + 1 < genes.size())
            s += ", ";
    }
    s += "]";
    return s;
}

// Helper to print coordinates
std::string pointToString(const Point& p) {
    std::stringstream ss;
    ss << "(" << std::fixed << std::setprecision(2) << p.x << ", " << p.y << ")";
    return ss.str();
}

// Dual output to console & file
struct DualLogger {
    std::ofstream file;
    explicit DualLogger(const std::string &fname) { file.open(fname, std::ios::app); }
    template <typename T> DualLogger &operator<<(const T &data) {
        std::cout << data;
        if (file.is_open())
            file << data;
        return *this;
    }
    typedef std::ostream &(*StreamManipulator)(std::ostream &);
    DualLogger &operator<<(StreamManipulator manip) {
        manip(std::cout);
        if (file.is_open())
            manip(file);
        return *this;
    }
};

// ------------------------------------------------------------
// Pre-computation Helpers
// ------------------------------------------------------------
struct PrecomputedData {
    std::vector<std::vector<double>> locDistance; // |L|×|L| matrix
    std::vector<std::vector<int>>    coverageLUT; // for each location, list of POI indices it covers
};

PrecomputedData precompute(const ProblemInstance &pb) {
    const size_t L = pb.locations.size();
    const size_t P = pb.pois.size();

    PrecomputedData pc;
    pc.locDistance.assign(L, std::vector<double>(L, 0.0));
    pc.coverageLUT.assign(L, {});

    // Distance matrix between locations
    for (size_t i = 0; i < L; ++i) {
        for (size_t j = i + 1; j < L; ++j) {
            double d = dist(pb.locations[i], pb.locations[j]);
            pc.locDistance[i][j] = pc.locDistance[j][i] = d;
        }
    }

    // Coverage LUT: which POIs each location covers
    for (size_t l = 0; l < L; ++l) {
        for (size_t p = 0; p < P; ++p) {
            if (dist(pb.locations[l], pb.pois[p]) <= pb.sensor.sense_range) {
                pc.coverageLUT[l].push_back(static_cast<int>(p));
            }
        }
    }
    return pc;
}

// ------------------------------------------------------------
// Fitness & Constraint Checks
// ------------------------------------------------------------
bool checkDuplicates(const Individual &ind) {
    std::set<int> s(ind.genes.begin(), ind.genes.end());
    return s.size() != ind.genes.size();
}

bool checkCoverage(const Individual &ind, const PrecomputedData &pc, size_t poiCount) {
    std::vector<bool> covered(poiCount, false);
    for (int locIdx : ind.genes) {
        for (int poiIdx : pc.coverageLUT[locIdx]) covered[poiIdx] = true;
    }
    return std::all_of(covered.begin(), covered.end(), [](bool v) { return v; });
}

bool checkConnectivity(const Individual &ind, const PrecomputedData &pc, double commRange) {
    const size_t N = ind.genes.size();
    if (N <= 1)
        return true; // single sensor trivially connected

    // Build adjacency list
    std::vector<std::vector<size_t>> adj(N);
    for (size_t i = 0; i < N; ++i) {
        int locI = ind.genes[i];
        for (size_t j = i + 1; j < N; ++j) {
            int locJ = ind.genes[j];
            if (pc.locDistance[locI][locJ] <= commRange) {
                adj[i].push_back(j);
                adj[j].push_back(i);
            }
        }
    }

    // BFS
    std::vector<bool> visited(N, false);
    std::queue<size_t> q;
    q.push(0);
    visited[0] = true;
    size_t count = 1;
    while (!q.empty()) {
        size_t u = q.front();
        q.pop();
        for (size_t v : adj[u]) {
            if (!visited[v]) {
                visited[v] = true;
                q.push(v);
                ++count;
            }
        }
    }
    return count == N;
}

void evaluateIndividual(Individual &ind, const PrecomputedData &pc, const ProblemInstance &pb) {
    bool dupl = checkDuplicates(ind);
    bool cov = !dupl && checkCoverage(ind, pc, pb.pois.size());
    bool conn = cov && checkConnectivity(ind, pc, pb.sensor.comm_range);
    ind.valid = (!dupl) && cov && conn;
    ind.fitness = ind.valid ? 1.0 : 0.0;
}

// ------------------------------------------------------------
// Genetic Operators
// ------------------------------------------------------------
Individual tournamentSelect(const std::vector<Individual> &pop, int t, std::mt19937 &rng) {
    std::uniform_int_distribution<int> distIdx(0, static_cast<int>(pop.size() - 1));
    int best = distIdx(rng);
    for (int i = 1; i < t; ++i) {
        int challenger = distIdx(rng);
        if (pop[challenger].fitness > pop[best].fitness)
            best = challenger;
    }
    return pop[best];
}

std::pair<Individual, Individual> uniformCrossover(const Individual &p1, const Individual &p2, double pcross, std::mt19937 &rng) {
    Individual c1 = p1; // Copy parent data initially
    Individual c2 = p2;
    c1.fitness = 0.0; // Reset fitness for children
    c2.fitness = 0.0;
    c1.valid = false;
    c2.valid = false;

    std::uniform_real_distribution<> dist(0.0, 1.0);
    if (dist(rng) < pcross) {
        std::uniform_int_distribution<> gene_dist(0, 1);
        for (size_t i = 0; i < p1.genes.size(); ++i) {
            if (gene_dist(rng) == 1) {
                std::swap(c1.genes[i], c2.genes[i]);
            }
        }
    }
    // Children C1, C2 might be identical to parents if crossover didn't happen
    return {c1, c2};
}

// Returns true if mutation occurred, false otherwise
bool mutateIndividual(Individual &ind, double pind, double pgene, int locationCount, std::mt19937 &rng, std::vector<int>& mutated_indices) {
    std::uniform_real_distribution<> dist(0.0, 1.0);
    mutated_indices.clear(); // Clear previous indices
    bool mutated = false;

    if (dist(rng) < pind) {
        std::uniform_int_distribution<> gene_idx_dist(0, static_cast<int>(ind.genes.size()) - 1);
        std::uniform_int_distribution<> loc_dist(0, locationCount - 1);

        for (size_t i = 0; i < ind.genes.size(); ++i) {
            if (dist(rng) < pgene) {
                ind.genes[i] = loc_dist(rng);
                mutated_indices.push_back(i);
                mutated = true;
            }
        }
    }
    return mutated;
}

// Prevent duplicates via simple repair (optional)
void repairDuplicates(Individual &ind, int locationCount, std::mt19937 &rng) {
    std::set<int> used;
    std::uniform_int_distribution<int> distLoc(0, locationCount - 1);
    for (int &g : ind.genes) {
        while (used.count(g)) {
            g = distLoc(rng);
        }
        used.insert(g);
    }
}

// ------------------------------------------------------------
// GA Core for a fixed N
// ------------------------------------------------------------
std::optional<Individual> runGA(int N, const ProblemInstance &pb, const PrecomputedData &pc, DualLogger &log) {
    const auto &gp = pb.gparams;
    int L = static_cast<int>(pb.locations.size());
    int P = pb.gparams.population_size;
    int T = pb.gparams.tournament_size;
    double PC = pb.gparams.prob_crossover;
    double PM = pb.gparams.prob_mutation;      // Individual probability
    double PGM = pb.gparams.prob_gene_mutation; // Gene probability
    int E = pb.gparams.num_elites;
    int MAX_GEN = pb.gparams.generations;
    int STAG_LIMIT = pb.gparams.stagnation_limit;

    // RNG setup
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<> dist_loc(0, L - 1);

    log << "\n================ Diagnostics AVANT la Génération 0 (N=" << N << ") ================" << std::endl;

    // Print distance matrix (partial)
    log << "Matrice de distance (5 premières lignes et colonnes) :\n";
    for (int i = 0; i < std::min(5, L); ++i) {
        for (int j = 0; j < std::min(5, L); ++j) {
            log << std::fixed << std::setprecision(2) << pc.locDistance[i][j] << " ";
        }
        log << "\n";
    }

    log << "Exigence de couverture : chaque POI doit être à moins de "
        << std::fixed << std::setprecision(2) << pb.sensor.sense_range
        << " unités d'au moins un capteur\n";
    log << "Codage entier ; longueur du chromosome = " << N << std::endl;
    for(int i=0; i < N; ++i) {
        log << "  Capteur " << i+1 << " -> indice de gène " << i << std::endl;
    }


    // 1. Initialize Population
    std::vector<Individual> population(P);
    log << "Population initiale (N=" << N << ") :\n";
    log << "Tentative de création de " << P << " individus via mélange...\n";

    // Create a list of all possible location indices
    std::vector<int> allLocationIndices(L);
    std::iota(allLocationIndices.begin(), allLocationIndices.end(), 0); // Fill with 0, 1, ..., L-1

    for (int i = 0; i < P; ++i) {
        log << "Création de l'individu " << i+1 << " (taille=" << N << ")..." << std::flush;

        // Shuffle the indices
        std::shuffle(allLocationIndices.begin(), allLocationIndices.end(), rng);

        // Assign the first N unique indices to the individual's genes
        population[i].genes.assign(allLocationIndices.begin(), allLocationIndices.begin() + N);
        assert(population[i].genes.size() == static_cast<size_t>(N));
        log << " [taille confirmée=" << population[i].genes.size() << "]" << std::flush;
        log << " Gènes assignés (uniques): " << genesToString(population[i].genes) << std::flush;
        evaluateIndividual(population[i], pc, pb);
        log << " Évalué. Fitness=" << std::fixed << std::setprecision(6) << population[i].fitness << std::endl;
    }
    log << "Population initiale créée avec succès." << std::endl;

    double bestFitnessOverall = 0.0;
    Individual bestIndividualOverall;
    int generations_without_improvement = 0;

    // GA Loop
    for (int gen = 0; gen < MAX_GEN; ++gen) {
        log << "\n================ Génération " << gen << " ================" << std::endl;

        // Find best fitness in current population
        double bestFitnessThisGen = 0.0;
        int bestIndexThisGen = -1;
        for(int i=0; i < P; ++i) {
             if (population[i].valid && population[i].fitness > bestFitnessThisGen) {
                 bestFitnessThisGen = population[i].fitness;
                 bestIndexThisGen = i;
             }
        }
        log << " Meilleur fitness de cette génération : " << std::fixed << std::setprecision(2) << bestFitnessThisGen << std::endl;

        // Check stagnation & update best overall
        if (bestFitnessThisGen > bestFitnessOverall) {
            bestFitnessOverall = bestFitnessThisGen;
            bestIndividualOverall = population[bestIndexThisGen]; // Store the best one
            generations_without_improvement = 0;
            log << " Nouveau meilleur fitness global : " << std::fixed << std::setprecision(2) << bestFitnessOverall << std::endl;
        } else {
            generations_without_improvement++;
        }

        if (bestFitnessOverall >= 1.0) {
            log << "\nSolution valide trouvée à la génération " << gen << "!" << std::endl;
            return bestIndividualOverall; // Found a valid solution
        }

        if (generations_without_improvement >= STAG_LIMIT) {
            log << "Stagnation atteinte à la génération " << gen << ". L'algorithme génétique s'arrête pour N=" << N << std::endl;
            return std::nullopt;
        }

        // Create next generation
        std::vector<Individual> next_population;
        next_population.reserve(P);

        // Elitism: Copy best E individuals directly
        std::sort(population.begin(), population.end(), [](const Individual &a, const Individual &b) {
            // Sort valid before invalid, then by fitness descending
            if (a.valid != b.valid) return a.valid > b.valid;
            return a.fitness > b.fitness;
        });
        for (int i = 0; i < E && i < P; ++i) {
            next_population.push_back(population[i]);
        }

        log << "\n--- Sélection, Croisement, Mutation --- " << std::endl;
        // Fill the rest of the population
        while (next_population.size() < P) {
            // Selection
            Individual p1 = tournamentSelect(population, T, rng);
            Individual p2 = tournamentSelect(population, T, rng);
            log << "Parents sélectionnés : P1= " << genesToString(p1.genes) << " (fitness=" << p1.fitness << ") P2= " << genesToString(p2.genes) << " (fitness=" << p2.fitness << ")" << std::endl;

            // Crossover
            auto children = uniformCrossover(p1, p2, PC, rng);
            log << " Enfants après croisement (avant mutation) : C1= " << genesToString(children.first.genes) << " C2= " << genesToString(children.second.genes) << std::endl;

            // Mutation
            std::vector<int> mutated_indices1, mutated_indices2;
            Individual c1_before_mutation = children.first;
            Individual c2_before_mutation = children.second;

            bool mutated1 = mutateIndividual(children.first, PM, PGM, L, rng, mutated_indices1);
            bool mutated2 = mutateIndividual(children.second, PM, PGM, L, rng, mutated_indices2);

            if(mutated1) {
                log << " Mutation de C1 : Avant= " << genesToString(c1_before_mutation.genes) << " Après= " << genesToString(children.first.genes) << " Indices=" << genesToString(mutated_indices1) << std::endl;
            } else {
                log << " Mutation de C1 : Aucune mutation." << std::endl;
            }
             if(mutated2) {
                log << " Mutation de C2 : Avant= " << genesToString(c2_before_mutation.genes) << " Après= " << genesToString(children.second.genes) << " Indices=" << genesToString(mutated_indices2) << std::endl;
            } else {
                log << " Mutation de C2 : Aucune mutation." << std::endl;
            }

            // Repair duplicates if needed
            repairDuplicates(children.first, L, rng);
            repairDuplicates(children.second, L, rng);

            // Evaluate new children
            evaluateIndividual(children.first, pc, pb);
            evaluateIndividual(children.second, pc, pb);

            next_population.push_back(children.first);
            if (next_population.size() < P)
                next_population.push_back(children.second);
        }

        population = std::move(next_population);

        // Log the new population
        log << "\nNouvelle population (Génération " << gen + 1 << ") taille=" << population.size() << ":" << std::endl;
        for (int i = 0; i < P; ++i) {
             log << " Individu #" << i << " gènes=" << genesToString(population[i].genes)
                 << " valide=" << (population[i].valid ? 'T' : 'F')
                 << " fitness=" << std::fixed << std::setprecision(2) << population[i].fitness << std::endl;
        }
    }

    log << "Nombre maximum de générations atteint. L'algorithme génétique s'arrête pour N=" << N << std::endl;
    return std::nullopt; // Failed to find solution within generation limit
}

// ------------------------------------------------------------
// JSON Parsing – can load one map or multiple maps in a single file
// ------------------------------------------------------------
// Forward declaration
ProblemInstance parseSingleMap(const json &j);

std::vector<ProblemInstance> loadProblems(const std::string &path) {
    std::ifstream fin(path);
    if (!fin) {
        throw std::runtime_error("Cannot open file: " + path);
    }
    json j;
    fin >> j;

    std::vector<ProblemInstance> problems;

    // Case 1: file itself is a single map object
    if (j.contains("locations") || j.contains("emplacements")) {
        problems.push_back(parseSingleMap(j));
        return problems;
    }

    // Case 2: top-level keys each map to a map object
    for (auto &el : j.items()) {
        if (el.value().is_object()
            && (el.value().contains("locations") || el.value().contains("emplacements"))
            && (el.value().contains("pois") || el.value().contains("points_interet"))) {
            problems.push_back(parseSingleMap(el.value()));
        }
    }
    if (problems.empty()) {
        throw std::runtime_error("No valid map objects with 'locations'/'emplacements' and 'pois'/'points_interet' found in JSON.");
    }
    return problems;
}

// ------------------ Parse one map object --------------------
ProblemInstance parseSingleMap(const json &jMap) {
    ProblemInstance pb;

    const json &j = jMap;

    // ---------------- LOCATIONS ----------------
    const json *locArrPtr = nullptr;
    if (j.contains("locations") && j.at("locations").is_array()) {
        locArrPtr = &j.at("locations");
    } else if (j.contains("emplacements") && j.at("emplacements").is_array()) {
        locArrPtr = &j.at("emplacements");
    } else {
        throw std::runtime_error("Map missing 'locations'/'emplacements' array");
    }
    // Locations
    for (const auto &pt : *locArrPtr) {
        if (pt.is_object() && pt.contains("x") && pt.contains("y") && pt.at("x").is_number() && pt.at("y").is_number()) {
            pb.locations.push_back({pt.at("x").get<double>(), pt.at("y").get<double>()});
        } else {
            throw std::runtime_error("Invalid location entry format: expected object with numeric 'x' and 'y'. Found: " + pt.dump());
        }
    }
    // ---------------- POIS ---------------------
    const json *poiArrPtr = nullptr;
    if (j.contains("pois") && j.at("pois").is_array()) {
        poiArrPtr = &j.at("pois");
    } else if (j.contains("points_interet") && j.at("points_interet").is_array()) {
        poiArrPtr = &j.at("points_interet");
    } else {
        throw std::runtime_error("Map missing 'pois'/'points_interet' array");
    }
    for (const auto &pt : *poiArrPtr) {
        if (pt.is_object() && pt.contains("x") && pt.contains("y") && pt.at("x").is_number() && pt.at("y").is_number()) {
            pb.pois.push_back({pt.at("x").get<double>(), pt.at("y").get<double>()});
        } else {
            throw std::runtime_error("Invalid POI entry format: expected object with numeric 'x' and 'y'. Found: " + pt.dump());
        }
    }
    // ---------------- SENSOR SPEC ---------------
    if (j.contains("sensors")) {
        const auto &s = j.at("sensors");
        if (s.is_object()) {
            if (s.contains("range") && s.at("range").is_number()) pb.sensor.sense_range = s.at("range").get<double>();
            if (s.contains("comm_range") && s.at("comm_range").is_number()) pb.sensor.comm_range = s.at("comm_range").get<double>();
        } else if (s.is_array() && s.size() >= 2 && s[0].is_number() && s[1].is_number()) {
            pb.sensor.sense_range = s[0].get<double>();
            pb.sensor.comm_range = s[1].get<double>();
        }
    } else if (j.contains("capteurs") && j.at("capteurs").is_array()) {
        const auto &caps = j.at("capteurs");
        if (!caps.empty() && caps[0].is_object() && caps[0].contains("rayon") && caps[0].at("rayon").is_number()) {
            pb.sensor.sense_range = caps[0].at("rayon").get<double>();
            pb.sensor.comm_range  = caps[0].at("rayon").get<double>();
        }
    }
    // ---------------- GENETIC PARAMS ------------
    if (j.contains("genetic_params")) {
        if (!j.at("genetic_params").is_object()) {
            std::cerr << "Warning: 'genetic_params' found but is not an object. Using defaults." << std::endl;
        } else {
            const auto &g = j.at("genetic_params");
            auto &gp = pb.gparams;

            // Helper lambda for safe numeric parsing
            auto safe_get_numeric = [&](const json& obj, const std::string& key, auto& target, const auto& default_val) {
                if (obj.contains(key)) {
                    const auto& val = obj.at(key);
                    if (val.is_number()) {
                        target = val.get<std::decay_t<decltype(target)>>();
                    } else {
                        std::cerr << "Warning: JSON key '" << key << "' is not a number. Using default: " << default_val << std::endl;
                        target = default_val; // Use default if type mismatch
                    }
                } else {
                    target = default_val; // Use default if key missing
                }
            };

            safe_get_numeric(g, "population_size", gp.population_size, gp.population_size);
            safe_get_numeric(g, "max_generations", gp.generations, gp.generations); // Note: struct field is 'generations'
            safe_get_numeric(g, "generations", gp.generations, gp.generations); // Alias for config1.json
            safe_get_numeric(g, "tournament_size", gp.tournament_size, gp.tournament_size);
            safe_get_numeric(g, "prob_crossover", gp.prob_crossover, gp.prob_crossover);
            safe_get_numeric(g, "crossover_rate", gp.prob_crossover, gp.prob_crossover); // Alias for config1.json
            safe_get_numeric(g, "prob_mutation", gp.prob_mutation, gp.prob_mutation);
            safe_get_numeric(g, "mutation_rate", gp.prob_mutation, gp.prob_mutation); // Alias for config1.json
            safe_get_numeric(g, "num_elites", gp.num_elites, gp.num_elites);

            // Nested prob_gene_mutation
            if (g.contains("mutation_types") && g.at("mutation_types").is_object()) {
                const auto& mutation_types = g.at("mutation_types");
                if (mutation_types.contains("classique") && mutation_types.at("classique").is_object()) {
                    const auto& classique = mutation_types.at("classique");
                    safe_get_numeric(classique, "prob_gene", gp.prob_gene_mutation, gp.prob_gene_mutation);
                } // else: classique missing or not object, keep default gp.prob_gene_mutation
            } // else: mutation_types missing or not object, keep default gp.prob_gene_mutation
        }
    }

    // Validation
    if (pb.locations.empty() || pb.pois.empty()) {
        throw std::runtime_error("JSON must contain non-empty locations and pois arrays.");
    }
    return pb;
}

// ------------------------------------------------------------
// Main
// ------------------------------------------------------------
int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " data.json" << std::endl;
        return 1;
    }

    const std::string jsonPath = argv[1];
    std::vector<ProblemInstance> problems;
    try {
        problems = loadProblems(jsonPath);
    } catch (const std::exception &ex) {
        std::cerr << "Error reading input: " << ex.what() << std::endl;
        return 1;
    }

    auto total_start_time = std::chrono::high_resolution_clock::now();
    int mapIndex = 0;

    for (auto &pb : problems) {
        auto map_start_time = std::chrono::high_resolution_clock::now();
        std::string results_filename = "results_map" + std::to_string(mapIndex) + ".txt";
        // Clear file content at the start for this map
        std::ofstream clear_file(results_filename, std::ios::trunc);
        clear_file.close();

        DualLogger log(results_filename);

        log << "============================================================\n";
        log << "Carte #" << mapIndex << " -- Emplacements: " << pb.locations.size() << ", POIs: " << pb.pois.size() << "\n";
        log << "Portée Capteur=" << pb.sensor.sense_range << ", Portée Comm=" << pb.sensor.comm_range << "\n";

        PrecomputedData pc = precompute(pb);

        // --- Impression des matrices initiales ---
        const size_t L = pb.locations.size();
        const size_t P_count = pb.pois.size();

        log << "\nMatrice de distance (Emplacement <-> POI) :\n";
        log << "        ";
        for (size_t p_idx = 0; p_idx < P_count; ++p_idx) {
            log << "POI " << std::setw(2) << p_idx << "   ";
        }
        log << "\n";
        for (size_t loc_idx = 0; loc_idx < L; ++loc_idx) {
            log << "Emp " << std::setw(3) << loc_idx << ": ";
            for (size_t p_idx = 0; p_idx < P_count; ++p_idx) {
                double d = dist(pb.locations[loc_idx], pb.pois[p_idx]);
                log << std::fixed << std::setprecision(6) << d << " ";
            }
            log << "\n";
        }
        log << "------------------------------------------------------------\n";

        log << "\nMatrice de couverture (Emplacement -> POI) pour portée = "
            << std::fixed << std::setprecision(6) << pb.sensor.sense_range << " :\n";
        log << "        ";
        for (size_t p_idx = 0; p_idx < P_count; ++p_idx) {
            log << "POI " << std::setw(2) << p_idx << " ";
        }
        log << "\n";
        for (size_t loc_idx = 0; loc_idx < L; ++loc_idx) {
            log << "Emp " << std::setw(3) << loc_idx << ": ";
            for (size_t p_idx = 0; p_idx < P_count; ++p_idx) {
                bool covers = false;
                for (int covered_poi : pc.coverageLUT[loc_idx]) {
                    if (covered_poi == static_cast<int>(p_idx)) { covers = true; break; }
                }
                log << "   " << (covers ? "1" : "0") << "   ";
            }
            log << "\n";
        }
        log << "============================================================\n";

        int maxN = static_cast<int>(pb.locations.size());
        int lastSuccessN = -1;
        Individual bestOverall;

        for (int N = 1; N <= maxN; ++N) {
            log << "\n============================================================" << std::endl;
            log << "Exécution de l'algorithme génétique pour N = " << N << " capteurs" << std::endl;
            auto solutionOpt = runGA(N, pb, pc, log);
            if (solutionOpt) {
                lastSuccessN = N;
                bestOverall = *solutionOpt;
                log << "\n‼️  SUCCÈS pour N = " << N << " (fitness=1)" << std::endl;
                break;
            } else {
                log << "\n❌  ÉCHEC pour N = " << N << std::endl;
            }
        }

        log << "\n================ RAPPORT FINAL ================\n";
        if (lastSuccessN != -1) {
            log << "Nombre minimum de capteurs N* = " << lastSuccessN << std::endl;
            log << "Gènes (Indices d'emplacement): " << genesToString(bestOverall.genes) << std::endl;
            log << "Coordonnées des capteurs déployés:\n";
            for(int gene_index : bestOverall.genes) {
                if (gene_index >= 0 && gene_index < pb.locations.size()) {
                    log << "  Index " << gene_index << ": " << pointToString(pb.locations[gene_index]) << "\n";
                } else {
                    log << "  Index invalide " << gene_index << " trouvé dans la solution!\n";
                }
            }
        } else {
            log << "Aucune solution valide trouvée pour aucun N.\n";
        }

        auto map_end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> map_duration = map_end_time - map_start_time;
        log << "\nDurée d'exécution pour la carte #" << mapIndex << ": " << std::fixed << std::setprecision(3) << map_duration.count() << " secondes\n";

        ++mapIndex;
    }

    auto total_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_duration = total_end_time - total_start_time;
    std::cout << "\nDurée d'exécution totale pour toutes les cartes: " << std::fixed << std::setprecision(3) << total_duration.count() << " secondes" << std::endl;

    return 0;
}
