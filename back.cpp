#include <openvdb/openvdb.h>
#include <openvdb/tools/ChangeBackground.h>
#include <iostream>
#include <fstream>

int main(int argc, char** argv) {
    assert(argc == 2);

    std::ifstream f(argv[1]);

    openvdb::initialize();
    openvdb::io::File file("out.vdb");
    openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create(0.);
    grid->setName("density");

    int min_x, min_y, min_z, max_x, max_y, max_z;
    f.read((char*)&min_x, 4);
    f.read((char*)&min_y, 4);
    f.read((char*)&min_z, 4);
    f.read((char*)&max_x, 4);
    f.read((char*)&max_y, 4);
    f.read((char*)&max_z, 4);
    printf("(%u, %u, %u) -> (%u, %u, %u)\n", min_x,min_y, min_z, max_x, max_y, max_z);

    auto accessor = grid->getAccessor();
    openvdb::Coord p;
    for (p.z() = min_z; p.z() <= max_z; p.z()++)
    {
        for (p.y() = min_y; p.y() <= max_y; p.y()++)
        {
            for (p.x() = min_x; p.x() <= max_x; p.x()++)
            {
                float v;
                f.read((char*)&v, 4);
                accessor.setValue(p, v);
            }
        }
    }
    file.write({grid});
    file.close();
}
