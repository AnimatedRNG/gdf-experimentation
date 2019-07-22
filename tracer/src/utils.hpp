#pragma once

#include <iostream>

#include "Halide.h"

#include <unordered_map>
#include <unordered_set>
#include <deque>
#include <tuple>
#include <string>

using namespace Halide;

typedef std::unordered_map<std::string, Func> parameter_map;

void wrap_children(std::vector<Internal::Function>& children,
                   std::unordered_map<std::string, Internal::Function>& substitutions,
                   const int& iteration) {
    std::unordered_map<std::string, Internal::Function> seen;
    std::deque<Internal::Function> agenda;
    agenda.clear();
    for (auto child : children) {
        std::cout << "child " << child.name() << std::endl;
        seen[child.name()] = child;
        agenda.push_back(child);
    }

    // Performs BFS on children
    while (!agenda.empty()) {
        Internal::Function& child = agenda.front();
        seen[child.name()] = child;
        agenda.pop_front();

        const std::string child_name(child.name());
        const std::map<std::string, Internal::Function> parents(
            Internal::find_direct_calls(child));

        for (auto& parent : parents) {
            std::cout << "parents of " << child_name << " include " << parent.first
                      << std::endl;
            if (substitutions.count(parent.first)) {
                std::cout << "In " << child.name() << " substituting "
                          << parent.second.name() << " with "
                          << substitutions[parent.first].name() << std::endl;
                child.substitute_calls(parent.second, substitutions[parent.first]);
            } else if (seen.count(parent.first) == 0) {
                agenda.push_back(parent.second);
            }
        }
    }
}

std::unordered_map<std::string, Func> wrap_reduction(
    std::function<std::unordered_map
    <std::string, Func>(std::unordered_map < std::string, Func>
                       ) > loop_body,
    std::unordered_map<std::string, Func> first_iteration,
    const int& iterations
) {
    std::unordered_map<std::string, Func> inputs(first_iteration);
    std::unordered_map<std::string, Func> output(loop_body(
                first_iteration));

    for (int i = 1; i < iterations; i++) {
        std::unordered_map<std::string, Func> new_inputs;

        for (auto input_pair : inputs) {
            if (output.count(input_pair.first)) {
                new_inputs[input_pair.first] = output[input_pair.first];
            } else {
                new_inputs[input_pair.first] = input_pair.second;
            }
        }

        output = loop_body(new_inputs);
        inputs = new_inputs;
    }

    return output;
}

inline void _print_func_dependencies(Internal::Function f,
                              std::unordered_map<std::string, Internal::Function>& seen) {
    auto mp = Internal::find_direct_calls(f);
    seen[f.name()] = f;
    if (mp.empty()) {
        std::cout << f.name() << " has no dependencies" << std::endl;
    } else {
        std::cout << f.name() << " depends on ";
        for (auto& pair : mp) {
            std::cout << pair.first << " ";
        }
        std::cout << std::endl;
        for (auto& pair : mp) {
            if (seen.count(pair.first) == 0) {
                _print_func_dependencies(pair.second, seen);
            }
        }
    }
}

inline void print_func_dependencies(Internal::Function f) {
    std::unordered_map<std::string, Internal::Function> seen;
    _print_func_dependencies(f, seen);
}

std::vector<Internal::Function> get_all_iterations(Func f) {
    std::deque<Internal::Function> agenda;
    agenda.clear();
    agenda.push_back(f.function());

    std::vector<Internal::Function> all_iterations;
    all_iterations.push_back(f.function());

    int iteration;
    std::string base_name;
    size_t mangle_index = f.name().find("$");
    if (mangle_index != std::string::npos) {
        iteration = std::stoi(f.name().substr(mangle_index + 1));
        base_name = f.name().substr(0, mangle_index);
        std::cout << "starting iteration is " << iteration << std::endl;
        iteration--;
    } else {
        std::cout << "can't even find first iteration" << std::endl;
        return all_iterations;
    }

    while (!agenda.empty()) {
        Internal::Function item = agenda.front();
        agenda.pop_front();
        auto mp = Internal::find_direct_calls(item);
        for (auto& child_entry : mp) {
            std::string child_name = child_entry.first;
            std::string iteration_name = base_name +
                                         std::string("$") + std::to_string(iteration);
            if (Internal::starts_with(child_name, iteration_name)) {
                all_iterations.push_back(child_entry.second);
                std::cout << "managed to find " << child_entry.first << std::endl;
                iteration -= 1;
            }

            agenda.push_back(child_entry.second);
        }
    }

    return all_iterations;
}

std::unordered_map<std::string, Func> create_example(
    std::unordered_map<std::string, Func> inp) {
    Func a("a"), b("b"), e("e");
    a() = inp["a"]();
    b() = a() * a();
    e() = b();

    return {{"b", b}, {"a", e}};
}
