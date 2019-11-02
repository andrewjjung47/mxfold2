#pragma once

#include <vector>
#include <string>
#include <tuple>
#include <variant>
#include <memory>
#include "trimatrix.h"

class Fold
{
    public:
        struct Options
        {
            size_t min_hairpin;
            size_t max_internal;
            std::string stru;
            bool use_penalty;
            std::string ref;
            float pos_paired;
            float neg_paired;
            float pos_unpaired;
            float neg_unpaired;

            Options() : 
                min_hairpin(3),
                max_internal(30),
                use_penalty(false)
            {    
            }

            Options& min_hairpin_loop_length(size_t s)
            {
                this->min_hairpin = s;
                return *this;
            }

            Options& max_internal_loop_length(size_t s)
            {
                this->max_internal = s;
                return *this;
            }

            Options& constraints(const std::string& s)
            {
                this->stru = s;
                return *this;
            }

            Options& penalty(const std::string& ref, float pos_paired=0, float neg_paired=0, float pos_unpaired=0, float neg_unpaired=0)
            {
                this->use_penalty = pos_paired!=0 || neg_paired!=0 || pos_unpaired!=0 || neg_unpaired!=0;
                this->ref = ref;
                this->pos_paired = pos_paired;
                this->neg_paired = neg_paired;
                this->pos_unpaired = pos_unpaired;
                this->neg_unpaired = neg_unpaired;
                return *this;
            }
        };

    public:
        static bool allow_paired(char x, char y);
        static auto parse_paren(const std::string& paren) 
            -> std::vector<u_int32_t>;
        static auto make_paren(const std::vector<u_int32_t>& p) -> std::string;
        static auto make_constraint(const std::string& seq, std::string stru, u_int32_t max_bp, bool canonical_only=true)
            -> std::pair<std::vector<std::vector<bool>>, std::vector<std::vector<bool>>>;
        static auto make_penalty(size_t L, bool use_penalty, const std::string& ref, float pos_paired, float neg_paired, float pos_unpaired, float neg_unpaired) 
            -> std::tuple<TriMatrix<float>, std::vector<std::vector<float>>, float>;
};